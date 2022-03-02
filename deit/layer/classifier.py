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
        self.distil_fc = nn.Linear(d_model, output_dim)

    def forward(self, img):
        emb = self.patch_embedding(img)
        for enc_layer in self.enc:
            emb = enc_layer(emb) # [batch_size, seq_length, d_model]

        cls_proj = self.fc(emb[:,0,:])
        distil_proj = self.distil_fc(emb[:,-1,:])
        return cls_proj, distil_proj
