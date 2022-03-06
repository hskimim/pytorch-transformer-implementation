from deit.patch_embedding.input_embedding import PatchEmbedding
from vision_transformer.layers.classifier import ViT
import torch
import torch.nn as nn

class DeiT(ViT) :
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
                 output_dim):

        super().__init__(height,
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
                 output_dim)

        self.patch_embedding = PatchEmbedding(
            height,
            width,
            channel,
            patch,
            d_model
        )
        self.distil_fc = nn.Linear(d_model, output_dim)

    def forward(self, img):
        emb = self.patch_embedding(img)
        for enc_layer in self.enc:
            emb = enc_layer(emb) # [batch_size, seq_length, d_model]

        cls_proj = self.fc(emb[:,0,:])
        distil_proj = self.distil_fc(emb[:,-1,:])

        return torch.log_softmax(cls_proj, dim=1) + torch.log_softmax(distil_proj, dim=1)
        # author added the softmax output by the two classifiers (cls_proj, distil_proj)
        # so we have to add the values at softmax scaled level not logit level
