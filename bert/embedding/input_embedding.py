import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(seq_length, d_model)
        self.seg_emb = nn.Embedding(3, d_model, padding_idx=pad_idx)

    def generate_enc_mask_m(self, src):
        mask_m = (src != 0).unsqueeze(1).unsqueeze(2)
        return mask_m

    def forward(self, txt, seg):
        emb = self.tok_emb(txt)
        pos = torch.arange(0, emb.shape[1]).unsqueeze(0).repeat(emb.shape[0], 1).to(emb.device)
        summed = emb + self.pos_emb(pos) + self.seg_emb(seg)
        return summed