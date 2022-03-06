import random
import torch

from vision_transformer.dataset.iterator import ImageNetIterator

class MaskedImageNetIterator(ImageNetIterator):
    def __init__(self,
                 root='../../imagenet/',
                 height=224,
                 width=224,
                 patch = 16,
                 mask_ratio=0.7,
                 is_train=True,
                 in_memory=False,
                 verbose=False):

        super().__init__(root,
                         height,
                         width,
                         is_train,
                         in_memory,
                         verbose)
        self.patch_size = patch ** 2
        self.mask_ratio = mask_ratio

    def mask_token(self, x):
        seq_length = x.shape[0]
        unmask_bool = torch.tensor([False] * seq_length)
        unmask_cnt = int(seq_length * (1 - self.mask_ratio))

        while unmask_bool.sum() < unmask_cnt:
            rand_idx = random.randrange(0, seq_length)
            unmask_bool[rand_idx] = True
        return x[unmask_bool], x[~unmask_bool], unmask_bool

    def __getitem__(self, idx):
        if not self.in_memory:
            x, label = self.xy_container[idx]
            img = self._convert_jpg_to_tensor(x)  # [c, h, w]
        else:
            img, label = self.cache_x[idx], self.cache_y[idx]

        C, H, W = img.shape

        splitted = img.view(C, -1).split(self.patch_size, -1)  # [C, H*W]

        stacked_tensor = torch.stack(splitted, dim=1)  # [C, (H*W)/(P**2), P**2]
        stacked_tensor = stacked_tensor.permute(1, 0, 2).contiguous()  # [(H*W)/(P**2), C, P**2]
        stacked_tensor = stacked_tensor.view(stacked_tensor.shape[0], -1)  # [(H*W)/(P**2), C * P**2]
        # sequence length : (H*W)/(P**2)

        unmask_tensor, mask_tensor, unmask_bool = self.mask_token(stacked_tensor)
        data = {
            'input' : unmask_tensor,
            'unmask_bool' : unmask_bool,
            'label' : mask_tensor,
        }
        return data
