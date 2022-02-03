from copy import deepcopy
import torch.nn as nn
from transformer.embedding.input_embedding import InputEmbedding
from bert.layer.encoder import EncoderLayer


class BERT(nn.Module):
    def __init__(self,
                 vocab_size,
                 seq_length,
                 pad_id,
                 d_model,
                 d_ff,
                 n_head,
                 dropout_p,
                 n_enc_layer):
        super().__init__()

        self.embber = InputEmbedding(vocab_size, seq_length, d_model, pad_id)

        enc = EncoderLayer(vocab_size, seq_length, d_model, d_ff, n_head, dropout_p)

        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])

    def forward(self, txt, seg):
        emb = self.embber(txt, seg)
        mask_m = self.embber.generate_enc_mask_m(txt)

        for enc_layer in self.enc:
            emb = enc_layer(emb, mask_m)

        return emb