from copy import deepcopy
import torch.nn as nn
import torch
from mae.layers.vit_block import EncoderBlock

class Decoder(nn.Module):

    def __init__(self,
                 d_model,
                 d_ff,
                 ffn_typ,
                 act_typ,
                 n_head,
                 seq_length,
                 dropout_p,
                 n_dec_layer):
        super().__init__()

        self.seq_length = seq_length
        self.mask_tok = nn.Paraameter(torch.randn(1, d_model), requires_grad=True)

        enc = EncoderBlock(
            d_model,
            d_ff,
            n_head,
            ffn_typ,
            act_typ,
            dropout_p
        )
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_dec_layer)])

    def forward(self, z, unmask_bool):
        mem = []
        cnt = 0
        for idx in unmask_bool:
            if idx.item() is True:  # unmask
                mem.append(z[:, cnt].unsqueeze(1))
                cnt += 1
            else:
                mem.append(self.mask_tok.unsqueeze(0).repeat(z.shape[0], 1, 1))
        z = torch.cat(mem, dim=1) # [batch-size, seq-length, hidden-dim]

        for enc_layer in self.enc:
            z = enc_layer(z)  # [batch_size, seq_length, d_model]
        return z
