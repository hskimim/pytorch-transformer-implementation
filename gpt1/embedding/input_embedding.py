import torch
import torch.nn as nn
import math

# newly implemented apart from transformer's since it is for variable length of sequence. (it is applied at mask_m generate part)

class InputEmbedding(nn.Module) :
    def __init__(self, vocab_size, max_length, d_model) :
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)

    def generate_dec_mask_m(self, src) :
        src_len = src.shape[1]
        src_sub_mask = torch.tril(torch.ones((src_len, src_len)), diagonal=0).bool().to(src.device) # mask subsequent token
        return self.convert_mask_to_sum_form(src_sub_mask)

    def convert_mask_to_sum_form(self, mask_m):
        sum_form = torch.empty_like(mask_m)
        sum_form[mask_m == 0] = float("-inf")
        sum_form[mask_m == 1] = 0
        return sum_form

    def forward(self, x) :
        emb = self.tok_emb(x)
        pos = torch.arange(0, emb.shape[1]).unsqueeze(0).repeat(emb.shape[0], 1).to(emb.device)
        summed = emb / math.sqrt(self.d_model) + self.pos_emb(pos)
        return summed