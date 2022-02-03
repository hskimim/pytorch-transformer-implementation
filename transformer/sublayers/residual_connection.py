import torch.nn as nn

class PostProcessing(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, emb, attn):
        return self.ln(emb + self.dropout(attn))