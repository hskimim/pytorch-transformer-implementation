import range as range
import torch
from .base import FactorizedAttentionMask

class FixedFA(FactorizedAttentionMask):
    """Fixed factorized attention"""
    def __init__(self, l, num_heads, c):
        super().__init__(l, num_heads)
        self.c = c

    def a_1(self, seq_len):  # A_{i}^(1)
        mask_m = self.init_mask(seq_len)
        for i in torch.arange(mask_m.shape[1]):
            start = i + 1
            step = (self.l + 1) - start % (self.l + 1)
            end = start + step
            if step < self.l + 1:
                end = min(mask_m.shape[1], end)
                tri_indices = torch.arange(start, end)
                mask_m[tri_indices, i] = 1
        return mask_m.bool()

    def a_2(self, seq_len):  # A_{i}^(2)
        mask_m = self.init_mask(seq_len)
        for i in torch.arange(mask_m.shape[0]):
            for j in torch.arange(mask_m.shape[1]):
                if j % self.l in torch.arange(self.l - self.c, self.l):
                    if j <= i :
                        mask_m[i, j] = 1
        return mask_m.bool()
