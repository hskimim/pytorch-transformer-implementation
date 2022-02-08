import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, pad_id):
        super().__init__()
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model // 6, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(seq_length, d_model // 6)
        self.seg_emb = nn.Embedding(3, d_model // 6, padding_idx=pad_id)

        # embedding matrix parameterization
        self.tok_proj = nn.Linear(d_model // 6, d_model)
        self.pos_proj = nn.Linear(d_model // 6, d_model)
        self.seg_proj = nn.Linear(d_model // 6, d_model)

        self.dp = nn.Dropout(0.1)

    def generate_enc_mask_m(self, src):
        mask_m = (src != 0).unsqueeze(1).unsqueeze(2)
        return self.convert_mask_to_sum_form(mask_m)

    def convert_mask_to_sum_form(self, mask_m):
        sum_form = torch.empty_like(mask_m)
        sum_form[mask_m == 0] = float("-inf")
        sum_form[mask_m == 1] = 0
        return sum_form

    def forward(self, txt, seg):
        emb = self.tok_emb(txt)
        pos = torch.arange(0, emb.shape[1]).unsqueeze(0).repeat(emb.shape[0], 1).to(emb.device)
        summed = self.tok_proj(emb) + self.pos_proj(self.pos_emb(pos)) + self.seg_proj(self.seg_emb(seg))
        return self.dp(summed)