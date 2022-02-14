import torch
import torch.nn as nn

from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from bert.sublayers.residual_connection import PostProcessing
from bert.sublayers.point_wise_ffn import PositionwiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout_p):
        super().__init__()

        self.ma = MultiHeadAttention(d_model, n_head)
        self.sdp = ScaledDotProductAttention(d_model)

        self.pp1 = PostProcessing(d_model, dropout_p)
        self.pp2 = PostProcessing(d_model, dropout_p)

        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff)

    def linear_proj(self, matrix, proj):
        """
        Random projection using the idea of JL lemma
        """
        # [batch-size, n-head, sequence-length, d-model] -> [batch-size, n-head, k, d-model]
        return torch.matmul(matrix.permute(0, 1, 3, 2).contiguous(), proj).permute(0, 1, 3, 2).contiguous()

    def forward(self, emb, mask_m, projector):
        q, k, v = self.ma(emb)
        k = self.linear_proj(k, projector.to(k.device))
        v = self.linear_proj(v, projector.to(k.device))

        attn = self.sdp(q, k, v, mask=mask_m)

        attn = self.pp1(emb, attn)
        z = self.positionwise_ffn(attn)

        return self.pp2(attn, z)
