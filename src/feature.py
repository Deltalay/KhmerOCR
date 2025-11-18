import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda"

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Learned2DPositionalEncoding(nn.Module):
    def __init__(self, H_patches, W_patches, dim):
        super().__init__()
        self.row_embed = nn.Parameter(torch.zeros(1, H_patches, 1, dim))
        self.col_embed = nn.Parameter(torch.zeros(1, 1, W_patches, dim))

    def forward(self, x):
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, D
        x = x + self.row_embed[:, :H, :, :] + self.col_embed[:, :, :W, :]
        x = x.view(B, H*W, D)
        return x

class CNNFeatureExtraction2D(nn.Module):
    def __init__(self, embed_dim=256, H_patches=8, W_patches=16):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-50 Bottleneck layers
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        # Projection to embedding dim
        self.proj = nn.Conv2d(2048, embed_dim, kernel_size=3, stride=1, padding=1)

        # Positional encoding
        self.pos_embed_2d = Learned2DPositionalEncoding(H_patches, W_patches, embed_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [Bottleneck(in_channels, out_channels, stride)]
        in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        x = self.pos_embed_2d(x)
        return x