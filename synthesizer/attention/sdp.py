import torch
import torch.nn as nn

class SynthesizedAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, B, v):
        scaled_score = torch.softmax(B, dim=-1)
        attention = torch.matmul(scaled_score, v).permute(0, 2, 3, 1).contiguous()
        attention = attention.view(attention.shape[0], attention.shape[1], self.d_model)
        return self.fc(attention)
