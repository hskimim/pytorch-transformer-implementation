import torch
import torch.nn as nn

class BertFC(nn.Module):
    def __init__(self, embedder, d_model, vocab_size):
        super().__init__()
        self.embedder = embedder
        self.mlm_fc = nn.Linear(d_model, vocab_size)

    def forward(self, txt, seg):
        emb = self.embedder(txt, seg)
        return torch.log_softmax(self.mlm_fc(emb), dim=-1), torch.log_softmax(self.nsp_fc(emb[:, 0]), dim=-1)
