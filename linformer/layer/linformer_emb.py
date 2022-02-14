from copy import deepcopy

import torch
import torch.nn as nn

from bert.embedding.input_embedding import InputEmbedding
from linformer.layer.encoder import EncoderLayer

class Linformer(nn.Module):
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

    def forward(self, txt, seg):
        emb = self.embber(txt, seg)

        for enc_layer in self.enc:
            emb = enc_layer(emb, None, self.EK)
            # Since k axis among [Q X K] is arbitrary aggregated. There is no mask

        return emb
