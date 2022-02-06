import torch.nn as nn

from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from transformer.sublayers.residual_connection import PostProcessing
from transformer.sublayers.point_wise_ffn import PositionwiseFFN

class MAB(nn.Module):
    """
    Multi-head Attention Block
    """
    def __init__(self, d_model, d_ff, n_head, dropout_p=0):
        super().__init__()

        self.ma1 = MultiHeadAttention(d_model, n_head)
        self.ma2 = MultiHeadAttention(d_model, n_head)
        self.sdp = ScaledDotProductAttention(d_model)

        self.pp1 = PostProcessing(d_model, dropout_p)
        self.pp2 = PostProcessing(d_model, dropout_p)

        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff)

    def forward(self, x, y, mask_m):
        q, k, v = self.ma1(x, y)
        attn = self.sdp(q, k, v, mask=mask_m)
        attn = self.pp1(x, attn)
        z = self.positionwise_ffn(attn)
        H = self.pp2(attn, z)
        return H
