import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.set_emb = nn.Linear(1, d_model) # projector for real-number

    def generate_enc_mask_m(self, input_set):
        mask_m = (input_set != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return self.convert_mask_to_sum_form(mask_m)

    def convert_mask_to_sum_form(self, mask_m):
        sum_form = torch.empty_like(mask_m)
        sum_form[mask_m == 0] = float("-inf")
        sum_form[mask_m == 1] = 0
        return sum_form

    def forward(self, x):
        # x : input_set, [batch_size, seq_length]
        emb = self.set_emb(x.unsqueeze(-1)) # [batch_size, seq_length, emb_dim]
        scaled_emb = emb / math.sqrt(self.d_model)
        return scaled_emb