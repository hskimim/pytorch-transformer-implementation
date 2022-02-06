import torch.nn as nn
from transformer.attention.sdp import ScaledDotProductAttention
from transformer.attention.mult_head_attn import MultiHeadAttention
from transformer.sublayers.residual_connection import PostProcessing
from transformer.sublayers.point_wise_ffn import PositionwiseFFN

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout_p):
        super().__init__()

        self.ma_self = MultiHeadAttention(d_model, n_head)
        self.ma_enc_dec = MultiHeadAttention(d_model, n_head)
        self.sdp_self = ScaledDotProductAttention(d_model)
        self.sdp_enc_dec = ScaledDotProductAttention(d_model)

        self.pp1 = PostProcessing(d_model, dropout_p)
        self.pp2 = PostProcessing(d_model, dropout_p)
        self.pp3 = PostProcessing(d_model, dropout_p)

        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff)

    def forward(self, emb, mask_m_src, mask_m_trg, enc_hidden):
        q, k, v = self.ma_self(emb)
        attn = self.sdp_self(q, k, v, mask=mask_m_trg)
        attn1 = self.pp1(emb, attn)

        dec_q, enc_k, enc_v = self.ma_enc_dec(attn1, enc_hidden)
        attn2 = self.sdp_enc_dec(dec_q, enc_k, enc_v, mask_m_src)
        sub_layer_output = self.pp2(attn1, attn2)

        z = self.positionwise_ffn(sub_layer_output)

        return self.pp3(sub_layer_output, z)
