import random
import torch

from vision_transformer.dataset.iterator import ImageNetIterator

class MaskedImageNetIterator(ImageNetIterator):
    def __init__(self,
                 root='../../imagenet/',
                 height=256,
                 width=256,
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
        self.mask_ratio = mask_ratio

    def mask_token(self, x):
        seq_length = x.shape[1]
        unmask_bool = torch.tensor([False] * seq_length)
        unmask_cnt = int(seq_length * (1 - self.mask_ratio))

        while unmask_bool.sum() < unmask_cnt:
            rand_idx = random.randrange(0, seq_length)
            unmask_bool[rand_idx] = True
        return x[:,unmask_bool], unmask_bool

    def __getitem__(self, idx):
        if not self.in_memory:
            x, y = self.xy_container[idx]
            img = self._convert_jpg_to_tensor(x)  # [c, h, w]
        else:
            img, label = self.cache_x[idx], self.cache_y[idx]

        N, C, H, W = img.shape

        splitted = img.view(N, C, -1).split(self.patch_size, -1)  # [N, C, H*W]
        stacked_tensor = torch.stack(splitted, dim=2)  # [N, C, (H*W)/(P**2), P**2]

        stacked_tensor = stacked_tensor.permute(0, 2, 1, 3).contiguous()  # [N, (H*W)/(P**2), C, P**2]
        stacked_tensor = stacked_tensor.view(N, stacked_tensor.shape[1], -1)  # [N, (H*W)/(P**2), C * P**2]
        # sequence length : (H*W)/(P**2)

        masked_tensor, unmask_bool = self.mask_token(stacked_tensor)
        data = {
            'image' : stacked_tensor,
            'mask_image' : masked_tensor,
            'unmask_bool' : unmask_bool,
            'label' : label,
        }
        return data
