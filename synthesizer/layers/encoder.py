import torch.nn as nn
from synthesizer.attention.factorized_dense import FactorizedDenseSynthesizer
from synthesizer.attention.random import RandomSynthesizer
from synthesizer.attention.sdp import SynthesizedAttention

from bert.sublayers.residual_connection import PostProcessing
from bert.sublayers.point_wise_ffn import PositionwiseFFN

class EncoderLayer(nn.Module):
    def __init__(self, d_model,
                 d_ff,
                 n_head,
                 seq_length,
                 factorized_shape,
                 k,
                 mixed_weight,
                 dropout_p):
        super().__init__()
        """
        mixed_weight : [dense's weight, random's weight]
        """

        assert hasattr(mixed_weight, '__iter__') and len(mixed_weight) == 2,\
            "mixed weight should be iterable whose length is 2"
        assert sum(mixed_weight) == 1, f"Total sum of weight should be 1, now is {sum(mixed_weight)}"

        self.mixed_weight = mixed_weight
        self.fds = None
        self.rs = None

        if mixed_weight[0] != 0 :
            self.fds = FactorizedDenseSynthesizer(d_model, n_head, seq_length, factorized_shape)
        if mixed_weight[1] != 0 :
            self.rs = RandomSynthesizer(d_model, n_head, seq_length, k)
        self.attn = SynthesizedAttention(d_model)

        self.pp1 = PostProcessing(d_model, dropout_p)
        self.pp2 = PostProcessing(d_model, dropout_p)

        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff)

    def forward(self, emb, mask_m):
        # generate synthetic attention matrix,
        # In mixed synthesizer mode, we're going to use V matrix from random synthesizer. there is no for calculating it
        if self.fds is not None :
            C, v = self.fds(emb)
        if self.rs is not None :
            R, v = self.rs(emb)

        # aggregate multiple synthetics
        if 0 not in self.mixed_weight :
            score = C * self.mixed_weight[0] + R * self.mixed_weight[1]
        elif self.mixed_weight[0] != 0 :
            score = C
        else :
            score = R

        # apply softmax to normalize synthetic matrix and dot product with v
        attn = self.attn(score, v, mask_m)

        attn = self.pp1(emb, attn)
        z = self.positionwise_ffn(attn)

        return self.pp2(attn, z)
