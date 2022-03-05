import torch.nn as nn
import torch
import numpy as np

class MaskToken(nn.Module) :
    def __init__(self,
                 seq_length,
                 mask_ratio):
        super().__init__()
        self.seq_length = seq_length
        self.mask_ratio = mask_ratio
        self.mask_length = seq_length - int(seq_length * mask_ratio)

    def forward(self, x):
        unmask_idx = np.random.randint(low=0, high=self.seq_length, size=self.mask_length)
        unmask_bool = torch.tensor([False] * self.seq_length)
        unmask_bool[unmask_idx] = True

        return x[:, unmask_bool, :], unmask_bool
