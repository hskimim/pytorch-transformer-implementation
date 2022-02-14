from copy import deepcopy

import torch
import torch.nn as nn

from bert.embedding.input_embedding import InputEmbedding
from linformer.layer.encoder import EncoderLayer

class Linformer(nn.Module):
    """
    The paper experimented using RoBERTa model with linformer's efficient attention mechanism,
    but in this implementation, I'm going to use bert training mechanism to re-use the existing module.
    """
    def __init__(self,
                 vocab_size,
                 seq_length,
                 pad_id,
                 d_model,
                 d_ff,
                 d_k,
                 n_head,
                 dropout_p,
                 n_enc_layer):
        super().__init__()

        self.embber = InputEmbedding(vocab_size, seq_length, d_model, pad_id)

        enc = EncoderLayer(d_model, d_ff, n_head, dropout_p)
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])

        self.EK = torch.randn(seq_length, d_k) / d_k # N(0, 1/k)
        # EK is linear project matrix for K,V and this module only supports
        # "head-wise sharing" and "key-value sharing" and "layer-wise sharing"
        # since there was no big difference in experiments from paper Figure (3)

    def forward(self, txt, seg):
        emb = self.embber(txt, seg)
        mask_m = self.embber.generate_enc_mask_m(txt)

        for enc_layer in self.enc:
            emb = enc_layer(emb, mask_m, self.EK)

        return emb
