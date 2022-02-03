import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, f"n_head({n_head}) does not divide d_model({d_model})"

        self.n_div_head = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)

    def div_and_sort_for_multiheads(self, projected, seq_len):
        div = projected.view(projected.shape[0], self.n_head, seq_len, self.n_div_head)
        return div

    def forward(self, emb, enc_inputs=None):
        seq_len = emb.shape[1]
        q = self.div_and_sort_for_multiheads(self.Q(emb), seq_len)
        k = self.div_and_sort_for_multiheads(self.K(emb), seq_len)
        v = self.div_and_sort_for_multiheads(self.V(emb), seq_len)

        return q, k, v