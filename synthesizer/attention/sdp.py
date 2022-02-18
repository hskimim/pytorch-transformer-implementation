import torch
import torch.nn as nn

class SynthesizedAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, score, v, mask):
        if mask is not None :
            score += mask.to(score.device)

        scaled_score = torch.softmax(score, dim=-1)
        attention = torch.matmul(scaled_score, v).permute(0, 2, 3, 1).contiguous()
        attention = attention.view(attention.shape[0], attention.shape[1], self.d_model)
        return self.fc(attention)
