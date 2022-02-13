import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_length, d_model, n_head, k):
        super().__init__()
        assert d_model % n_head == 0, f"n_head({n_head}) does not divide d_model({d_model})"

        self.seq_length = seq_length
        self.n_div_head = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head
        self.k = k # TODO : supports appropriate k for make linformer to have linear complexity

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.E = nn.Linear(seq_length, k)
        self.F = nn.Linear(seq_length, k)

        self.F.weight.requires_grad = False
        self.F.bias.requires_grad = False
        self.F.weight.requires_grad = False
        self.F.bias.requires_grad = False

    def div_and_sort_for_multiheads(self, projected, seq_len):
        div = projected.view(projected.shape[0], self.n_head, seq_len, self.n_div_head)
        # [batch-size, n-heads, seq-length,d-model/n-heads]
        return div

    def forward(self, emb, enc_inputs=None):
        seq_len = emb.shape[1]
        q = self.div_and_sort_for_multiheads(self.Q(emb), seq_len)

        if enc_inputs is not None:  # enc-dec attention
            seq_len = enc_inputs.shape[1]  # takes target sequence length for k and v
            k = self.div_and_sort_for_multiheads(self.K(enc_inputs), seq_len)
            v = self.div_and_sort_for_multiheads(self.V(enc_inputs), seq_len)

        else:  # self-attention
            k = self.div_and_sort_for_multiheads(self.K(emb), seq_len)
            v = self.div_and_sort_for_multiheads(self.V(emb), seq_len)

        # random projection to low dimensionality
        # [batch-size, n-heads, k, d-model/n-heads]
        k = self.E(k.permute(0,1,3,2).contiguous()).permute(0,1,3,2) # (E*K*W^K)
        v = self.F(v.permute(0,1,3,2).contiguous()).permute(0,1,3,2) # (K*V*W^V)
        return q, k, v
