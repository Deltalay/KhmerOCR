import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Multi-Head Attention for ViT
# -------------------------------
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.01, proj_drop=0.01):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------------
# ViT Encoder Block
# -------------------------------
class ViTEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTMultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------
# CNN Feature Extractor (ResNet-50)
# -------------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        base = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        # Keep up to the last conv layer
        self.features = nn.Sequential(*list(base.children())[:-2])
        # Project 2048 -> embed_dim
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)

    def forward(self, x):
        # Output shape: (B, embed_dim, H/32, W/32)
        x = self.features(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        # Flatten to sequence (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)


# -------------------------------
# KhmerOCRViT (ResNet + ViT)
# -------------------------------
class KhmerOCRViT(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, depth=6, num_classes=128, pretrained=True):
        super().__init__()
        self.cnn = ResNetFeatureExtractor(embed_dim, pretrained=pretrained)

        # Learnable 2D positional embedding
        # Initialize later based on input feature map size
        self.pos_embed = None

        self.encoder = nn.ModuleList([ViTEncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x, (H, W) = self.cnn(x)  # (B, N, D), (H_feat, W_feat)
        B, N, D = x.shape

        # Initialize learnable position embedding if not yet created
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            self.pos_embed = nn.Parameter(torch.zeros(1, N, D, device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        x = x + self.pos_embed
        for blk in self.encoder:
            x = blk(x)

        x = self.head(x)
        return x

if __name__ == "__main__":
    model = KhmerOCRViT(embed_dim=1024, num_classes=2048).to(device)
    dummy = torch.randn(2, 3, 224, 448).to(device)  # example OCR input (H, W)
    out = model(dummy)
    print("Output shape:", out.shape)
