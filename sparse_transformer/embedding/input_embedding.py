import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_length, d_model)

    def generate_attn_mat(self, seq_len, l):
        """
        prepare stride position index for generating stride aggregated position embedding
        """
        attn_mat = torch.empty((l, seq_len)).int()
        for idx in torch.arange(l):
            pos = torch.arange(0 - idx, seq_len - idx).unsqueeze(0)
            attn_mat[idx] = pos
        attn_mat[attn_mat < 0] = 0
        return attn_mat

    def forward(self, x, l):
        emb = self.tok_emb(x)

        attn_mat = self.generate_attn_mat(emb.shape[1], l).to(emb.device)
        pos_emb = self.pos_emb(attn_mat).sum(0).unsqueeze(0)
        pos_emb = pos_emb.repeat(emb.shape[0], 1, 1)

        summed = emb / math.sqrt(self.d_model) + pos_emb
        return summed
