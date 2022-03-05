import torch.nn as nn
from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from transformer.sublayers.point_wise_ffn import PositionwiseFFN

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 n_head,
                 ffn_typ,
                 act_typ,
                 dropout_p):
        super().__init__()

        self.ma = MultiHeadAttention(d_model, n_head)
        self.sdp = ScaledDotProductAttention(d_model)

        self.ln = nn.LayerNorm(d_model)
        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff, ffn_typ, act_typ)

        self.dp1 = nn.Dropout(dropout_p)
        self.dp2 = nn.Dropout(dropout_p)

    def forward(self, emb):

        # z_{l} = MSA((LN(z_{l-1}))+z_{l-1})
        q, k, v = self.ma(self.dp1(emb))
        attn = self.sdp(q, k, v, mask=None)

        # z_{l} = MLP((LN(z_{l-1}))+z_{l-1})
        z = self.positionwise_ffn(self.ln(self.dp2(attn))) + emb

        return z
