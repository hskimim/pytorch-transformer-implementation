from copy import deepcopy
import torch.nn as nn
from vision_transformer.patch_embedding.input_embedding import PatchEmbedding
from vision_transformer.layers.encoder import EncoderLayer
class ViT(nn.Module):
    def __init__(self,
                 height,
                 width,
                 channel,
                 patch,
                 d_model,
                 d_ff,
                 n_head,
                 dropout_p,
                 n_enc_layer,
                output_dim):

        super().__init__()
        self.patch_embedding = PatchEmbedding(
            height,
            width,
            channel,
            patch,
            d_model
        )
        enc = EncoderLayer(
            d_model,
            d_ff,
            n_head,
            dropout_p
        )
        self.enc = nn.ModuleList([deepcopy(enc) for _ in range(n_enc_layer)])
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, img):
        emb = self.patch_embedding(img)
        for enc_layer in self.enc:
            emb = enc_layer(emb) # [batch_size, seq_length, d_model]

        return self.fc(emb[:,0,:])
