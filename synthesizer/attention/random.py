import torch
import torch.nn as nn

class RandomSynthesizer(nn.Module):
    def __init__(self, d_model, n_head, n, k=None):
        super().__init__()
        assert d_model % n_head == 0, f"n_head({n_head}) does not divide d_model({d_model})"

        self.n_div_head = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head

        # randomly initialized matrix R_{h,l}
        if k is None :
            self.R = nn.Parameter(torch.randn(n_head, n, n), requires_grad=True)
        else :
            # factorization is applied O(n**2) -> O(2*n*K)
            a = nn.Parameter(torch.randn(n_head, n, k), requires_grad=True)
            b = nn.Parameter(torch.randn(n_head, n, k), requires_grad=True)
            self.R = torch.matmul(a, b.permute(0,2,1).contiguous())

        self.V = nn.Linear(d_model, d_model)

    def div_and_sort_for_multiheads(self, projected, seq_len):
        div = projected.view(projected.shape[0], seq_len, self.n_head, self.n_div_head)
        # [batch-size, n-heads, seq-length,d-model/n-heads]
        return div

    def forward(self, emb, enc_inputs=None):
        seq_len = emb.shape[1]

        if enc_inputs is not None:  # enc-dec attention
            seq_len = enc_inputs.shape[1]  # takes target sequence length for k and v
            v = self.div_and_sort_for_multiheads(self.V(enc_inputs), seq_len)
        else:  # self-attention
            v = self.div_and_sort_for_multiheads(self.V(emb), seq_len)
        v = v.transpose(0,2,1,3).contiguous()
        # v : [batch_size, n_head, seq_len, d_model//n_head]
        return self.R, v
