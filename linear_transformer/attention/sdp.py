import torch
import torch.nn as nn

from linear_transformer.feature_map.kernels import EluFeatureMap

class ScaledDotProductAttention(nn.Module):
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

        # apply mask to k
        if mask is not None:  # TODO : make attention mask be matched with k's dimension
            k += mask.to(k.device)

        deno = 1 / (torch.einsum('bhne,bhne->bhn', q, k.cumsum(2)) + self.eps)  # [batch-size, n-heads, seq-length]
        # k.cumsum(2) = Z, since z_{i} = z_{i-1} + pi(x_{i} * W_{K} (=projection of K)

        s = torch.einsum('bhne,bhnv->bhnev', k, v)  # outer product, pi(K) * V.T
        num = torch.einsum('bhne,bhnev->bhnv', q, s.cumsum(2))  # [batch-size, n-heads, seq-length, d_model//h-heads]
        # s_{i} = s_{i-1} + pi(x_{i} * W_{k}) * (x_{i} * W_{v})

        attention = num / deno.unsqueeze(-1)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(attention.shape[0], attention.shape[1], self.d_model)  # concat the multi-heads
        return self.fc(attention)
