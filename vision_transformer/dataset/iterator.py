from glob import glob
import os
import itertools
import warnings
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageNetIterator(Dataset):
    def __init__(self,
                 root='../../imagenet/',
                 height=256,
                 width=256,
                 is_train=True,
                 in_memory=False,
                 verbose=False):

        self.root = os.path.join(root, 'train') if is_train else os.path.join(root, 'valid')
        self.h = height
        self.w = width
        self.in_memory = in_memory
        self._load_fnames()
        if in_memory:
            self._load_all_caches(verbose)

    def _load_fnames(self):
        labels = sorted(list(map(lambda x: os.path.basename(x), glob(os.path.join(self.root, '*')))))
        self.label_dict = {l: li for li, l in enumerate(labels)}

        xy_container = []

        for l in labels:
            fnames = sorted(glob(os.path.join(self.root, l, '*')))
            xy_container += list(zip(fnames, itertools.repeat(self.label_dict[l])))

        self.xy_container = xy_container

    def _load_all_caches(self, verbose):
        warnings.warn("In-memory version, watch out your RAM overload!")
        ls = self.xy_container if not verbose else tqdm(self.xy_container, desc='caching...')
        tmp_x = []
        tmp_y = []
        for x, y in ls:
            tensor = self._convert_jpg_to_tensor(x).unsqueeze(0)
            tmp_x.append(tensor)
            tmp_y.append(y)
        self.cache_x = torch.cat(tmp_x, dim=0)
        self.cache_y = torch.tensor(tmp_y)

    def _convert_jpg_to_tensor(self, fname):
        img = Image.open(fname).convert('RGB')

        resize = transforms.Resize([self.h, self.w])
        img = resize(img)

        to_tensor = transforms.ToTensor()
        tensor = to_tensor(img)
        tensor = tensor
        return tensor

    def __len__(self):
        return len(self.xy_container)

    def __getitem__(self, idx):
        if not self.in_memory:
            x, y = self.xy_container[idx]
            x = self._convert_jpg_to_tensor(x)  # [c, h, w]
            return x, y
        else:
            return self.cache_x[idx], self.cache_y[idx]
