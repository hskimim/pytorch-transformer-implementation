import torch.nn as nn
from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from transformer.sublayers.residual_connection import PostProcessing
from bert.sublayers.point_wise_ffn import PositionwiseFFN

class Resblock(nn.Module):
    """
    Resblock(h),
        normalizes the input to the attention block and position-wise feedforward network
    """
    def __init__(self, d_model, d_ff, n_head, dropout_p):
        super().__init__()

        self.ma = MultiHeadAttention(d_model, n_head)
        self.sdp = ScaledDotProductAttention(d_model)

        self.ln = nn.LayerNorm(d_model)
        self.norm = PostProcessing(d_model, dropout_p)

        self.ff = PositionwiseFFN(d_model, d_ff)

        self.dp1 = nn.Dropout(dropout_p)
        self.dp2 = nn.Dropout(dropout_p)

    def forward(self, h, mask_m):

        h = self.ln(h)
        q, k, v = self.ma(h)
        a = self.sdp(q, k, v, mask=mask_m) # a(H)

        a = self.norm(h, self.dp1(a))
        b = self.dp2(self.ff(a)) # b(H)

        return a + b # a(H) + b(H)
