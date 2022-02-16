import torch.nn as nn

class EluFeatureMap(nn.Module) :
    def __init__(self):
        super().__init__()
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(x) + 1
