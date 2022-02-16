from linear_transformer.attention.sdp import LinearScaledDotProductAttention
from transformer.seq2seq.encoder import EncoderLayer as QuadraticEncoderLayer

class EncoderLayer(QuadraticEncoderLayer):
    def __init__(self, d_model, d_ff, n_head, dropout_p):
        super().__init__(d_model, d_ff, n_head, dropout_p)
        self.sdp = LinearScaledDotProductAttention(d_model)
