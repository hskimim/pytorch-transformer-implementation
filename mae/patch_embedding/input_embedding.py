import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 d_model):
        super().__init__()
        self.d_model = d_model

        self.patch_size = patch ** 2
        img_size = height * width
        assert img_size % self.patch_size == 0, 'img is not divisible with patch'

        self.seq_length = img_size // self.patch_size
        input_dim = self.patch_size * channel
        self.patch_emb = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(self.seq_length + 1, d_model)

        self.ln = nn.LayerNorm(d_model)

    def forward(self, masked_tensor):
        batch_size, seq_length, _ = masked_tensor.shape

        projected = self.patch_emb(masked_tensor)  # [N, sequence length, d_model]

        pos = torch.arange(0, seq_length)\
            .unsqueeze(0).repeat(batch_size, 1).to(projected.device)

        summed = projected + self.pos_emb(pos)  # [N, sequence length * mask_ratio, d_model]

        return self.ln(summed) # according to paper, input of MSA is normalized with layer axis
