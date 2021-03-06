import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        score = torch.matmul(q, k.permute(0, 1, 3, 2).contiguous()) / math.sqrt(self.d_model)

        if mask is not None :
            if mask.dtype == torch.bool :
                score = score.masked_fill(mask == 0, float('-inf'))
            elif mask.dtype == torch.float :
                score += mask.to(score.device)
            else :
                msg = f"Check your dtype of attetion mask. allowed dtype is {'bool', 'float'} but {mask.dtype}"
                raise TypeError(msg)

        scaled_score = torch.softmax(score, dim=-1)
        attention = torch.matmul(scaled_score, v).permute(0, 2, 3, 1).contiguous()
        attention = attention.view(attention.shape[0], attention.shape[1], self.d_model)
        return self.fc(attention)
