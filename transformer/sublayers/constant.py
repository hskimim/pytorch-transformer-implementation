import torch.nn as nn

class NoPreparedActError(Exception) :
    pass

class NoPreparedFFNError(Exception) :
    pass

act_container = \
    {
        'Linear' : nn.Identity(),
        'ReLU': nn.RELU(),
        'GELU' : nn.RELU(),
        'Swish' : nn.SiLU(),
}
