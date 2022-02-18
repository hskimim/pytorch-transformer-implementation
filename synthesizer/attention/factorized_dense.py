import torch.nn as nn
from math import prod

from synthesizer.attention.dense import DenseSynthesizer

class FactorizedDenseSynthesizer(nn.Module) :
    def __init__(self, d_model,
                 n_head,
                 seq_length,
                 factorized_shape=None):
        super().__init__()

        # TODO : erase below assertions using advanced type annotation
        if factorized_shape is not None :
            assert hasattr(factorized_shape, "__iter__") and len(factorized_shape) == 2, "The length factorized shape should be 2"
            msg = f"multiplication of {factorized_shape[0]} X {factorized_shape[1]} should be {seq_length}"
            assert prod(factorized_shape) == seq_length, msg
            self.fds1 = DenseSynthesizer(d_model,n_head, factorized_shape[0])
            self.fds2 = DenseSynthesizer(d_model,n_head, factorized_shape[1])
        else :
            self.fds = DenseSynthesizer(d_model, n_head, seq_length)
        self.factorized_shape = factorized_shape

    def forward(self, emb, enc_inputs=None):
        if self.factorized_shape is not None :
            A,_ = self.fds1(emb, enc_inputs)
            A = A.repeat(1,1,1,self.factorized_shape[1])

            B,v = self.fds2(emb, enc_inputs)
            B = B.repeat(1,1,1,self.factorized_shape[0])

            C = A * B
        else :
            C,v = self.fds(emb, enc_inputs)

        return C, v
