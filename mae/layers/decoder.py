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
                 dropout_p,
                 n_dec_layer):
        super().__init__()

        self.mask_tok = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)

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
        unshuffled = self.mask_tok.repeat(unmask_bool.shape[0], unmask_bool.shape[1], 1) # [batch-size, seq-length, d-model]
        unshuffled[unmask_bool] = z.view(-1, z.shape[-1])

        for enc_layer in self.enc:
            unshuffled = enc_layer(unshuffled)  # [batch_size, seq_length, d_model]
        return unshuffled
