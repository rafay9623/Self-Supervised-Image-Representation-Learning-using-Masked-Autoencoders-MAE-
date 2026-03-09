import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm       = self.norm1(x)
        attn_out, _  = self.attn(x_norm, x_norm, x_norm)
        x            = x + attn_out
        x            = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask_tokens_indices=None):
        if x.dim() == 4:
            x = self.patch_embed(x)

        if mask_tokens_indices is not None:
            if isinstance(mask_tokens_indices, np.ndarray):
                mask_tokens_indices = torch.from_numpy(mask_tokens_indices).long()
            elif not isinstance(mask_tokens_indices, torch.Tensor):
                mask_tokens_indices = torch.tensor(mask_tokens_indices, dtype=torch.long)
            mask_tokens_indices = mask_tokens_indices.to(self.pos_embed.device)

            B         = x.shape[0]
            pos_embed = self.pos_embed.expand(B, -1, -1)
            idx       = mask_tokens_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            pos_embed = torch.gather(pos_embed, dim=1, index=idx)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_channels=3,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        )

    def forward(self, x, mask_tokens_indices):
        return self.vit(x, mask_tokens_indices=mask_tokens_indices)


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=196, patch_size=16, embed_dim=768,
                 depth=12, num_heads=6, decoder_embed_dim=384):
        super().__init__()
        self.num_patches       = num_patches
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder_embed     = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token        = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.pred_head    = nn.Linear(decoder_embed_dim, patch_size * patch_size * 3)

    def forward(self, encoder_latent, visible_indices):
        B = encoder_latent.shape[0]
        x = self.decoder_embed(encoder_latent)

        if isinstance(visible_indices, np.ndarray):
            visible_indices = torch.from_numpy(visible_indices).long()
        elif not isinstance(visible_indices, torch.Tensor):
            visible_indices = torch.tensor(visible_indices, dtype=torch.long)
        visible_indices = visible_indices.to(x.device)
        if visible_indices.dim() == 1:
            visible_indices = visible_indices.unsqueeze(0).expand(B, -1)

        mask_tokens = (
            self.mask_token
            .expand(B, self.num_patches, -1)
            .to(dtype=x.dtype)
        )
        idx      = visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        full_seq = mask_tokens.scatter(dim=1, index=idx, src=x)

        full_seq = full_seq + self.decoder_pos_embed
        for block in self.decoder_blocks:
            full_seq = block(full_seq)

        full_seq = self.decoder_norm(full_seq)
        return self.pred_head(full_seq)


class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mask_ratio=0.75,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6):
        super().__init__()

        self.img_size    = img_size
        self.patch_size  = patch_size
        self.mask_ratio  = mask_ratio
        num_patches      = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.num_visible = int(num_patches * (1 - mask_ratio))
        self.num_masked  = num_patches - self.num_visible

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, encoder_embed_dim)
        self.encoder     = MAEEncoder(
            img_size=img_size, patch_size=patch_size,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
        )
        self.decoder     = MAEDecoder(
            num_patches=num_patches, patch_size=patch_size,
            embed_dim=encoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, decoder_embed_dim=decoder_embed_dim,
        )

    def patchify(self, x):
        p = self.patch_size
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

    def unpatchify(self, x):
        p     = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        return rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h, w=w, p1=p, p2=p, c=3)

    def random_masking(self, x):
        B           = x.shape[0]
        noise       = torch.rand(B, self.num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones(B, self.num_patches, device=x.device)
        mask[:, self.num_visible:] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        ids_keep  = ids_shuffle[:, :self.num_visible]
        x_visible = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )
        return x_visible, mask, ids_keep

    def forward(self, x):
        patches                          = self.patch_embed(x)
        x_visible, mask, visible_indices = self.random_masking(patches)
        encoder_latent                   = self.encoder(x_visible, visible_indices)
        pred                             = self.decoder(encoder_latent, visible_indices)
        return pred, mask

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss   = (pred - target) ** 2
        loss   = loss.mean(dim=-1)
        loss   = (loss * (1 - mask)).sum() / (1 - mask).sum()
        return loss