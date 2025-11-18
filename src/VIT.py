import torch
import torch.nn as nn
import torch.nn.functional as F
from feature import CNNFeatureExtraction2D
device = "cuda"
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.01, proj_drop=0.01):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape 
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_drop.p, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
class ViTEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, 
                 attn_drop=0., proj_drop=0., drop=0.):
        """
        dim        : embedding dimension
        num_heads  : number of attention heads
        mlp_ratio  : hidden size of the MLP = mlp_ratio * dim
        qkv_bias   : use bias in QKV linear layers
        attn_drop  : dropout on attention weights
        proj_drop  : dropout after output projection
        drop       : dropout after MLP
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTMultiHeadAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Multi-Head Self-Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x
class KhmerOCRViT(nn.Module):
    def __init__(self, embed_dim=256, H_patches=16, W_patches=32, num_heads=8, depth=6, num_classes=128):
        super().__init__()
        self.cnn = CNNFeatureExtraction2D(embed_dim, H_patches, W_patches)
        self.encoder = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.cnn(x)         # (B, N, embed_dim)
        for block in self.encoder:
            x = block(x)        # (B, N, embed_dim)
        x = self.head(x)        # (B, N, num_classes)
        return x