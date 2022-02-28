import torch.nn as nn
import torch

from set_transformer.layer.mab import MAB

class ISAB(nn.Module):
    """
    Induced Set Attention Block
    """
    def __init__(self, d_model, d_ff, n_head, ffn_typ, act_typ, induc_p=4, dropout_p=0):
        super().__init__()

        assert induc_p, "Only ISAB (induc_p > 0) is supported"
        self.I = nn.Parameter(torch.randn(induc_p, d_model).unsqueeze(0))

        self.mab1 = MAB(d_model, d_ff, n_head, ffn_typ, act_typ, dropout_p)
        self.mab2 = MAB(d_model, d_ff, n_head, ffn_typ, act_typ, dropout_p)

    def forward(self, emb, mask_m):
        # H = MAB(I, X) -> [m,d]
        H = self.mab1(self.I, emb, mask_m)

        # Z = MAB(X, H) -> [n,d]
        Z = self.mab2(emb, H, mask_m)
        return Z