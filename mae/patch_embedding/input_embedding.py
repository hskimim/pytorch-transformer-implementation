import torch
import torch.nn as nn
from mae.patch_embedding.mask_token import MaskToken

class PatchEmbedding(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 mask_ratio,
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

        self.masker = MaskToken(self.seq_length, mask_ratio)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img):
        N, C, H, W = img.shape

        splitted = img.view(N, C, -1).split(self.patch_size, -1)  # [N, C, H*W]
        stacked_tensor = torch.stack(splitted, dim=2)  # [N, C, (H*W)/(P**2)`, P**2]

        stacked_tensor = stacked_tensor.permute(0, 2, 1, 3).contiguous()  # [N, (H*W)/(P**2), C, P**2]
        stacked_tensor = stacked_tensor.view(N, stacked_tensor.shape[1], -1)  # [N, (H*W)/(P**2), C * P**2]
        # sequence length : (H*W)/(P**2)

        projected = self.patch_emb(stacked_tensor)  # [N, sequence length, d_model]
        remained_projected, unmask_idx = self.masker(projected)
        pos = torch.arange(0, self.masker.mask_length + 1).unsqueeze(0).repeat(N, 1).to(projected.device)
        summed = remained_projected + self.pos_emb(pos)  # [N, sequence length * mask_ratio, d_model]

        return self.ln(summed) # according to paper, input of MSA is normalized with layer axis
