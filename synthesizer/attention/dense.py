import torch.nn as nn

class DenseSynthesizer(nn.Module):
    def __init__(self, d_model,
                 n_head,
                 seq_length):

        super().__init__()
        assert d_model % n_head == 0, f"n_head({n_head}) does not divide d_model({d_model})"

        self.n_div_head = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head

        self.W1 = nn.Linear(self.n_div_head, self.n_div_head)
        self.W2 = nn.Linear(self.n_div_head, seq_length)
        self.act = nn.ReLU()
        self.V = nn.Linear(d_model, d_model)

    def div_and_sort_for_multiheads(self, projected, seq_len):
        div = projected.view(projected.shape[0], seq_len, self.n_head, self.n_div_head)
        # [batch-size, n-heads, seq-length,d-model/n-heads]
        return div

    def forward(self, emb, enc_inputs=None):
        seq_len = emb.shape[1]
        head_div_emb = self.div_and_sort_for_multiheads(emb, seq_len) # [batch_size, seq_len, n_head, d_model//n_head]

        # synthesize B_{i,h,l}
        B = self.W2(self.act(self.W1(head_div_emb))).transpose(0,2,1,3).contiguous()
        # [batch_size, n_head, seq_len, {seq_len}], {.} is hyper-parameter

        if enc_inputs is not None:  # enc-dec attention
            seq_len = enc_inputs.shape[1]  # takes target sequence length for k and v
            v = self.div_and_sort_for_multiheads(self.V(enc_inputs), seq_len)
        else:  # self-attention
            v = self.div_and_sort_for_multiheads(self.V(emb), seq_len)
        v = v.transpose(0,2,1,3).contiguous()
        # v : [batch_size, n_head, seq_len, d_model//n_head]
        return B, v
