import torch.nn as nn

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
                 n_head,
                 dropout_p,
                 n_enc_layer,
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
            dropout_p,
            n_dec_layer
        )
        self.fc = nn.Linear(d_model, channel * patch ** 2)

    def forward(self, mask_img, unmask_bool):
        encoded = self.enc(mask_img)
        decoded = self.dec(encoded, unmask_bool)

        masked_z = decoded[~unmask_bool]
        masked_z = masked_z.view(unmask_bool.shape[0], -1, masked_z.shape[-1]) # [batch_size, seq_length * mask_ratio, d_model]

        return self.fc(masked_z) # [batch_size, seq_length * mask_ratio, C * P ** 2]
