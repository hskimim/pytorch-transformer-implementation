import torch
import torch.nn as nn

from linear_transformer.feature_map.kernels import EluFeatureMap

class LinearScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)
        self.pi = EluFeatureMap()
        self.eps = 1e-5

    def forward(self, q, k, v, mask):
        # apply kernel, dimensionality is same since it uses elu feature map
        q = self.pi(q)
        k = self.pi(k)

        deno = 1 / (torch.einsum('bhne,bhne->bhn', q, k.cumsum(2)) + self.eps)  # [batch-size, n-heads, seq-length]
        # k.cumsum(2) = Z, since z_{i} = z_{i-1} + pi(x_{i} * W_{K} (=projection of K)

        seq_length = q.shape[2]
        mem = []
        for idx in range(seq_length): # TODO : accelerate the loop operation
            s_i = torch.einsum('bhe,bhv->bhev', k[:, :, :idx + 1].sum(2), v[:, :, :idx + 1].sum(2))
            # s_{i} = s_{i-1} + pi(x_{i} * W_{k}) * (x_{i} * W_{v})^{T}
            num_i = torch.einsum('bhe,bhev->bhv', q[:, :, idx], s_i).unsqueeze(2)
            mem.append(num_i)
        num = torch.cat(mem, dim=2)

        attention = num / deno.unsqueeze(-1)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(attention.shape[0], attention.shape[1], self.d_model)  # concat the multi-heads
        return self.fc(attention)
