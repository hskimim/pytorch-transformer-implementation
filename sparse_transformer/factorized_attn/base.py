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

    def a_1(self, seq_len):  # A_{i}^(1)
        pass

    def a_2(self, seq_len):  # A_{i}^(1)
        pass

    def generate_multi_head_attn_mask(self, seq_len):
        a1 = self.a_1(seq_len)
        a2 = self.a_2(seq_len)
        multi_head_mask_m = torch.empty(size=(1, self.num_heads, seq_len, seq_len)).bool()
        for idx in torch.arange(self.num_heads):
            if idx % 2 == 0:
                multi_head_mask_m[:, idx] = a1
            else:
                multi_head_mask_m[:, idx] = a2
        return multi_head_mask_m
