import torch.nn as nn

class PostProcessing(nn.Module):
    def __init__(self, d_model, p=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, emb, attn):
        return emb + self.dropout(self.ln(attn))
        # the location of layer normalization is different with transformers'
        # and it affects the performance of bert training a lot. needs to research on it