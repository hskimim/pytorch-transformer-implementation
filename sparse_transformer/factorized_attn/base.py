import torch
import torch.nn as nn

class FactorizedAttentionMask(nn.Module):
    """base-module for factorized attention modules"""
    def __init__(self, l, num_heads):
        super().__init__()
        self.l = l
        self.num_heads = num_heads

    def init_mask(self, seq_len):
        mat = torch.zeros(seq_len, seq_len)
        mat[torch.arange(mat.shape[0]), torch.arange(mat.shape[1])] = 1  # attend to diag
        return mat

    def mask_triu(self, mask_m):
        triu_indices = torch.triu_indices(*mask_m.shape, offset=1)
        mask_m[triu_indices[0], triu_indices[1]] = 0
        return mask_m

    def a_1(self, seq_len):  # A_{i}^(1)
        mat = torch.ones(seq_len, seq_len)
        return self.convert_mask_to_sum_form(self.mask_triu(mat)) # default is full-attention (gpt1 style)

    def a_2(self, seq_len):  # A_{i}^(1)
        mat = torch.ones(seq_len, seq_len)
        return self.convert_mask_to_sum_form(self.mask_triu(mat)) # default is full-attention

    def generate_multi_head_attn_mask(self, seq_len):
        a1 = self.a_1(seq_len)
        a2 = self.a_2(seq_len)
        multi_head_mask_m = torch.empty(size=(1, self.num_heads, seq_len, seq_len))#.bool()
        for idx in torch.arange(self.num_heads):
            if idx % 2 == 0:
                multi_head_mask_m[:, idx] = a1
            else:
                multi_head_mask_m[:, idx] = a2
        return multi_head_mask_m

    def convert_mask_to_sum_form(self, mask_m):
        sum_form = torch.empty_like(mask_m)
        sum_form[mask_m == 0] = float("-inf")
        sum_form[mask_m == 1] = 0
        return sum_form