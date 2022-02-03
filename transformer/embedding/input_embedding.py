import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, pad_id):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(seq_length, d_model)

    def generate_enc_mask_m(self, src):
        mask_m = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        return mask_m

    def generate_dec_mask_m(self, trg):
        trg_pad_mask = (trg != self.pad_id).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len)), diagonal=0).bool().to(trg_pad_mask.device)
        mask_m = trg_pad_mask & trg_sub_mask
        return mask_m

    def forward(self, x):
        emb = self.tok_emb(x)
        pos = torch.arange(0, emb.shape[1]).unsqueeze(0).repeat(emb.shape[0], 1).to(emb.device)
        summed = emb / math.sqrt(self.d_model) + self.pos_emb(pos)
        return summed