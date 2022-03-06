import torch.nn as nn

from mae.patch_embedding.input_embedding import PatchEmbedding
from mae.layers.encoder import Encoder
from mae.layers.decoder import Decoder

class MAE(nn.Module) :
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 d_model,
                 d_ff,
                 ffn_typ,
                 act_typ,
                 mask_ratio,
                 n_head,
                 dropout_p,
                 n_enc_layer,
                 seq_length,
                 n_dec_layer
                 ):
        super().__init__()

        self.enc = Encoder(
            height,
            width,
            channel,
            patch,
            d_model,
            d_ff,
            ffn_typ,
            act_typ,
            mask_ratio,
            n_head,
            dropout_p,
            n_enc_layer
        )

        self.dec = Decoder(
            d_model,
            d_ff,
            ffn_typ,
            act_typ,
            n_head,
            seq_length,
            dropout_p,
            n_dec_layer
        )

    def forward(self, mask_img, unmask_bool):
        encoded = self.enc(mask_img)
        decoded = self.dec(encoded, unmask_bool)
        masked_z = decoded[:, ~unmask_bool]
        return masked_z
