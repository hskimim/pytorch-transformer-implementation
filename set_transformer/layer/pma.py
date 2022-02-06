import torch
import torch.nn as nn

from set_transformer.layer.mab import MAB
from transformer.sublayers.point_wise_ffn import PositionwiseFFN

class PMA(nn.Module):
    """
    Pooling by Multihead Attention
    """
    def __init__(self, d_model, d_ff, n_head, k=4, dropout_p=0):
        super().__init__()

        self.S = nn.Parameter(torch.randn(k, d_model).unsqueeze(0))

        self.mab1 = MAB(d_model, d_ff, n_head, dropout_p)
        self.mab2 = MAB(d_model, d_ff, n_head, dropout_p)
        self.r_ff = PositionwiseFFN(d_model, d_ff)

    def forward(self, Z, mask_m):
        pooled = self.mab1(self.S, Z, mask_m) # [k,d]
        # omit the feed-forward layer in the beginning of the decoder which follows experiments detail of paper

        H = self.mab2(pooled, pooled, mask_m) # [k,d]
        return H
