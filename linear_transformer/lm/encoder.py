from copy import deepcopy
import torch.nn as nn
from gpt1.embedding.input_embedding import InputEmbedding
from linear_transformer.lm.enc_block import EncoderLayer

class LM(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_length,
                 d_model,
                 d_ff,
                 n_head,
                 dropout_p,
                 n_enc_layer):
        super().__init__()

        self.src_embber = InputEmbedding(vocab_size, max_length, d_model)
        enc = EncoderLayer(d_model, d_ff, n_head, dropout_p)
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src_emb = self.src_embber(src)

        for enc_layer in self.enc:
            src_emb = enc_layer(src_emb, mask_m=None)
            # we going to use full-attention since it already has auto-regressive characteristic like RNN
        return self.fc(src_emb)
