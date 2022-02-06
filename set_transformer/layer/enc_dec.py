from copy import deepcopy
import torch.nn as nn

from set_transformer.embeddings.input_embedding import InputEmbedding
from set_transformer.layer.isab import ISAB
from set_transformer.layer.pma import PMA

class EncoderDecoder(nn.Module):

    def __init__(self,
                 d_model,
                 d_ff,
                 n_head,
                 induc_p,
                 k,
                 pad_idx,
                 output_dim,
                 dropout_p,
                 n_enc_layer):

        super().__init__()

        self.embedder = InputEmbedding(d_model, pad_idx)
        enc = ISAB(d_model, d_ff, n_head, induc_p, dropout_p)
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])
        self.dec = PMA(d_model, d_ff, n_head, k, dropout_p)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, input_set):
        # input_set : [batch_size, seq_length(padded as 0)]

        emb = self.embedder(input_set)
        mask_m = self.embedder.generate_enc_mask_m(input_set)

        for enc_layer in self.enc:
            emb = enc_layer(emb, mask_m)

        z = self.dec(emb, mask_m=None) # there is no mask for pooled attention matrix
        return self.fc(z.squeeze())
