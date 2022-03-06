from glob import glob
import os
import itertools
import warnings
from tqdm import tqdm
import torch
import torchvision.models as models
from vision_transformer.dataset.iterator import ImageNetIterator

class DistilationImageNetIterator(ImageNetIterator):
    def __init__(self,
                 root='../../imagenet/',
                 height=224,
                 width=224,
                 is_train=True,
                 in_memory=False,
                 verbose=False,
                 teacher=models.resnet18(True),
                 distil_method='hard',
                 device=None):

        super().__init__(root,
                         height,
                         width,
                         is_train,
                         in_memory,
                         verbose)

        assert distil_method in {'hard', 'soft'}, "distil_method in {'hard', 'soft'}"
        self.distil_method = distil_method
        if device is None :
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.teacher = teacher
        self.teacher.to(device)
        self.teacher.eval()

    def __getitem__(self, idx):
        if not self.in_memory:
            x, y = self.xy_container[idx]
            x = self._convert_jpg_to_tensor(x)  # [c, h, w]
        else:
            x, y = self.cache_x[idx], self.cache_y[idx]

        predict = self.teacher(x.unsqueeze(0).to(self.device))

        if self.distil_method == 'hard':
            d = predict.argmax(1)
        else:
            d = torch.softmax(predict, dim=1)

        return {
            'input' : x,
            'label' : y,
            'distil_label' : d,
        }
