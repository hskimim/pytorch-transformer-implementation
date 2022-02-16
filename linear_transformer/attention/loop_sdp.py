import torch
import torch.nn as nn

from linear_transformer.feature_map.kernels import EluFeatureMap

# un-parallelize version of scaled dot product

class LoopScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)
        self.pi = EluFeatureMap()
        self.eps = 1e-5

    def init_s_v(self, batch_size, n_head, k_dim, v_dim):
        self.s = nn.Parameter(torch.zeros(batch_size, n_head, k_dim), requires_grad=True)
        self.v = nn.Parameter(torch.zeros(batch_size, n_head, k_dim, v_dim), requires_grad=True)

    def update_s(self, step, s):
        k_i = self.k[:,:,step,:] # [batch-size, n-head, k_dim]
        v_i = self.v[:,:,step,:] # [batch-size, n-head, v_dim]
        kvt = torch.einsum('bhk,bhv->bhkv', k_i, v_i) # [batch-size, n-head, k_dim, v_dim]
        s += kvt

    def update_z(self, step, z): # TODO : it can be replaced with cumsum operation
        k_i = self.k[:,:,step,:] # [batch-size, n-head, k_dim]
        z += k_i

    def forward(self, q, k, v):
        # apply kernel, dimensionality is same since it uses elu feature map
        seq_len = q.shape[2]
        q = self.pi(q)
        k = self.pi(k)

        # initialize s_{0} and k_{0}
        self.init_s_v(q.shape[0], q.shape[1], k.shape[-1], v.shape[-1])

        for idx in torch.arange(1, seq_len) : # TODO : address in-place operation issues...
            self.update_z(step=idx, z=self.z) # [batch-size, n-head, k_dim]
            self.update_s(step=idx, s=self.s) # [batch-size, n-head, k_dim, v_dim]
            q_i = q[:,:,idx,:] # [batch-size, n-head, q_dim]
            deno_i = 1 / (torch.einsum('nhe,nhe->nh', q_i, self.z).shape + self.eps)
            no_i = torch.einsum('nhe,nhev->nhv', q_i, self.s)
            v_prime_i = no_i / deno_i.unsqueeze(-1) # [batch-size, n-head, v_dim]

        return