import torch
import torch.nn as nn
from transformer.sublayers.constant import *

"""
follows "Noam Shazeer's GLU Variants Improve Transformer, (2020) https://arxiv.org/pdf/2002.05202.pdf
"""

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, ffn_typ='ffn', act_typ='ReLu'):
        super().__init__()

        if ffn_typ.lower() == 'ffn' :
            ff = FFN
        elif ffn_typ.lower() == 'glu' :
            ff = GLUFFN
        else :
            raise NoPreparedFFNError("ffn_typ should be between {'ffn', 'glu'}")

        act = act_container.get(act_typ, None)
        if act is None :
            raise NoPreparedActError(f"act_typ should be between {act_container.keys()}")
        self.ffn = ff(d_model, d_ff, act)

    def forward(self, x):
        return self.ffn(x)

class FFN(nn.Module) :
    def __init__(self, d_model,
                 d_ff,
                 act=nn.RELU()):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = act

    def forward(self, x) :
        return self.fc2(torch.relu(self.fc1(x)))

# gated linear units and variants
class GLUFFN(FFN) :
    def __init__(self, d_model,
                 d_ff,
                 act=nn.RELU()):
        super().__init__(d_model, d_ff, act)
        self.fc3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gated = self.act(self.fc1(x)) * self.fc2(x)
        projected = self.fc3(gated)
        return projected
