from copy import deepcopy
import torch.nn as nn

from mae.patch_embedding.input_embedding import PatchEmbedding
from mae.layers.vit_block import EncoderBlock

class Encoder(nn.Module):
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
                 n_enc_layer):

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            height,
            width,
            channel,
            patch,
            d_model
        )
        enc = EncoderBlock(
            d_model,
            d_ff,
            n_head,
            ffn_typ,
            act_typ,
            dropout_p
        )
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])

    def forward(self, masked_img):
        emb = self.patch_embedding(masked_img)

        for enc_layer in self.enc:
            emb = enc_layer(emb) # [batch_size, seq_length, d_model]

        return emb
