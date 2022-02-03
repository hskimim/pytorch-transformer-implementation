import torch.nn as nn
from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from bert.sublayers.residual_connection import PostProcessing
from bert.sublayers.point_wise_ffn import PositionwiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, seq_length, d_model, d_ff, n_head, dropout_p):
        super().__init__()

        self.ma = MultiHeadAttention(d_model, seq_length, n_head)
        self.sdp = ScaledDotProductAttention(d_model)

        self.pp1 = PostProcessing(d_model, dropout_p)
        self.pp2 = PostProcessing(d_model, dropout_p)

        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff)

    def forward(self, emb, mask_m):
        q, k, v = self.ma(emb)
        attn = self.sdp(q, k, v, mask=mask_m)

        attn = self.pp1(emb, attn)
        z = self.positionwise_ffn(attn)

        return self.pp2(attn, z)
