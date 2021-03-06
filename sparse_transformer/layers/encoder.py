from copy import deepcopy

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from sparse_transformer.embedding.input_embedding import InputEmbedding
from sparse_transformer.layers.resblock import Resblock
from sparse_transformer.factorized_attn.base import FactorizedAttentionMask as FullFA
from sparse_transformer.factorized_attn.fixed import FixedFA
from sparse_transformer.factorized_attn.strided import StridedFA

class Encoder(nn.Module):
    """
    pre-activation residual block
    """
    def __init__(self,
                 vocab_size,
                 max_length,
                 d_model,
                 d_ff,
                 n_head,
                 dropout_p,
                 attn_typ,
                 l,
                 c,
                 n_enc_layer):
        super().__init__()

        self.embeddings = InputEmbedding(vocab_size, max_length, d_model)
        enc = Resblock(d_model, d_ff, n_head, dropout_p)
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])
        self.final_act = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))
        self.seq_len = max_length
        self.l = l

        if attn_typ == 'fixed' :
            self.fa = FixedFA(l, n_head, c)
        elif attn_typ == 'strided' :
            self.fa = StridedFA(l, n_head)
        elif attn_typ == 'full' :
            self.fa = FullFA(l, n_head)
        else :
            raise ValueError("Not implemented yet")
        self.mask_m = self.fa.generate_multi_head_attn_mask(seq_len=max_length)

    def forward(self, src):
        h = self.embeddings(src, self.l)

        if h.shape[1] != self.seq_len :
            mask_m = self.fa.generate_multi_head_attn_mask(seq_len=h.shape[1])
        else :
            mask_m = self.mask_m

        for enc_layer in self.enc:
            h = h + enc_layer(h, mask_m)

        return checkpoint(self.final_act, h)
