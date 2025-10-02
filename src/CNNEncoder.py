import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from crop import prepare_crops_for_cnn  # your YOLO + pHash cropper
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --------------------------
# CNN Encoder
# --------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoderVisualize(nn.Module):
    def __init__(self, d_model=256, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, d_model, kernel_size=(7,1), stride=1)
        self.bn3 = nn.BatchNorm2d(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H=128, W_var]
        Returns:
            x_seq: [B, W_seq, D] - sequence embeddings
            x1, x2, x3: intermediate feature maps
        """
        x1 = F.relu(self.bn1(self.conv1(x)))   # First block
        x2 = F.relu(self.bn2(self.conv2(x1)))  # Second block
        x3 = F.relu(self.bn3(self.conv3(x2)))  # Third block

        # Global avg pool over height, keep width as sequence
        x_seq = x3.mean(2)           # [B, D, W3]
        x_seq = x_seq.permute(0,2,1) # [B, W3, D]

        return x_seq, x1, x2, x3
class CNNEncoder(nn.Module):
    """Extract feature embeddings from crops."""
    def __init__(self, d_model=256, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, d_model, 7, 1)  # reduce HxW -> 1x1 if needed
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(d_model)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(2)  # [B, D, H*W]
        x = x.mean(-1)    # global avg pool -> [B, D]
        return x

# --------------------------
# Run feature extraction
# --------------------------
def extract_features(image_path, target_size=224, d_model=256):
    crops = prepare_crops_for_cnn(image_path, target_size=target_size)  # [num_crops, 3, H, W]
    if crops.shape[0] == 0:
        print("No crops detected.")
        return None

    encoder = CNNEncoder(d_model=d_model, in_channels=3).eval()
    with torch.no_grad():
        features = encoder(crops)  # [num_crops, d_model]

    return features

# --------------------------
# Visualize features
# --------------------------
def visualize_features(features, method='pca'):
    """
    Visualize feature embeddings in 2D.
    method: 'pca' or 'tsne'
    """
    features_np = features.cpu().numpy()
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=3, random_state=42)

    feats_2d = reducer.fit_transform(features_np)

    plt.figure(figsize=(6,6))
    plt.scatter(feats_2d[:,0], feats_2d[:,1], c='blue', alpha=0.7)
    plt.title(f"Feature Embeddings ({method.upper()})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()
def visualize_feature_maps(feature_maps, num_channels=2):
    """
    feature_maps: torch.Tensor [B, C, H, W]
    num_channels: how many channels to show per block
    """
    fmap = feature_maps[0].cpu().detach()  # take first image
    C = fmap.shape[0]

    plt.figure(figsize=(15, 3))
    for i in range(min(num_channels, C)):
        plt.subplot(1, num_channels, i+1)
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis('off')
        plt.title(f"Ch {i}")
    plt.show()