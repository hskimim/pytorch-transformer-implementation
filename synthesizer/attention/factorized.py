import torch
import torch.nn as nn
from math import prod

from synthesizer.attention.dense import DenseSynthesizer
from synthesizer.attention.random import RandomSynthesizer

class FactorizedDenseSynthesizer(nn.Module) :
    def __init__(self, d_model,
                 n_head,
                 seq_length,
                 synthesizer_type='dense',
                 factorized_shape=None):
        super().__init__()

        # TODO : erase below assertions using advanced type annotation
        assert synthesizer_type in {'dense', 'random'}, "synthsizer should be between {'dense', 'random'}"
        if factorized_shape is not None :
            assert type(factorized_shape) == type((1,))
            assert len(factorized_shape) == 2, "The length factorized shape should be 2"
            msg = f"multiplication of {factorized_shape[0]} X {factorized_shape[1]} should be {seq_length}".format()
            assert prod(factorized_shape) == seq_length, msg

        self.synthesizer_dict = {
            'dense' : DenseSynthesizer(d_model, n_head, seq_length),
            'random' : RandomSynthesizer(d_model, n_head, seq_length),
        }

    def forward(self):
        pass
