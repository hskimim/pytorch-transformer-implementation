import range as range
import torch
from .base import FactorizedAttentionMask

class StridedFA(FactorizedAttentionMask):
    """strided factorized attention"""

    def a_1(self, seq_len):  # A_{i}^(1)
        mask_m = self.init_mask(seq_len)
        for i in range(mask_m.shape[1]):
            start = i + 1
            end = start + self.l
            if end > mask_m.shape[0]:
                end = mask_m.shape[0]

            prev_indices = torch.arange(start, end)
            mask_m[prev_indices, i] = 1
        return mask_m.bool()

    def a_2(self, seq_len):  # A_{i}^(2)
        mask_m = self.init_mask(seq_len)
        for i in range(mask_m.shape[1] - self.l - 1):
            start = i + 1 + self.l
            mod_indices = torch.arange(start, mask_m.shape[0], l)
            mask_m[mod_indices, i] = 1
        return mask_m.bool()