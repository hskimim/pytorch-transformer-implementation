from copy import deepcopy
import torch.nn as nn
from transformer.embedding.input_embedding import InputEmbedding
from transformer.seq2seq.encoder import EncoderLayer
from transformer.seq2seq.decoder import DecoderLayer

class EncoderDecoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_seq_length,
                 trg_seq_length,
                 src_pad_id,
                 trg_pad_id,
                 d_model,
                 d_ff,
                 n_head,
                 dropout_p,
                 n_enc_layer,
                 n_dec_layer):

        super().__init__()

        self.src_embber = InputEmbedding(src_vocab_size, src_seq_length, d_model, src_pad_id)
        self.trg_embber = InputEmbedding(trg_vocab_size, trg_seq_length, d_model, trg_pad_id)

        enc = EncoderLayer(d_model, d_ff, n_head, dropout_p)
        dec = DecoderLayer(trg_seq_length, d_model, d_ff, n_head, dropout_p)

        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])
        self.dec = nn.ModuleList([deepcopy(dec) for _ in range(n_dec_layer)])

        self.fc = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg):

        src_emb, trg_emb = self.src_embber(src), self.trg_embber(trg)
        src_mask_m = self.src_embber.generate_enc_mask_m(src)
        trg_mask_m = self.trg_embber.generate_dec_mask_m(trg)

        for enc_layer in self.enc:
            src_emb = enc_layer(src_emb, src_mask_m)

        for dec_layer in self.dec:
            trg_emb = dec_layer(trg_emb, src_mask_m, trg_mask_m, src_emb)

        return self.fc(trg_emb)
