import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """A minimal CNN encoder that converts cropped images into embeddings."""

    def __init__(self, d_model: int = 256, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, d_model, (8, 1), (8, 1))
        self.bn1, self.bn2, self.bn3 = (
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Embeddings of shape [B, T, D].
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.squeeze(2)  # [B, D, T]
        x = x.permute(0, 2, 1)  # [B, T, D]
        return x


class CropDataset(Dataset):
    """Dataset for cropped images, with resizing, padding, and mask generation."""

    def __init__(self, img_dir: str, target_height: int = 32, max_width: int = 128) -> None:
        self.paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
        self.target_height = target_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        scale = self.target_height / h
        new_w = min(int(w * scale), self.max_width)
        img = cv2.resize(img, (new_w, self.target_height))

        canvas = 255 * np.ones((self.target_height, self.max_width), dtype=np.uint8)
        canvas[:, :new_w] = img

        tensor = torch.from_numpy(canvas).float().unsqueeze(0) / 255.0
        mask = torch.zeros(self.max_width, dtype=torch.bool)
        mask[:new_w] = 1
        return tensor, mask, path


def collate(batch):
    """Collate function for DataLoader."""
    imgs, masks, paths = zip(*batch)
    imgs = torch.stack(imgs)  # [B, 1, H, W]
    masks = torch.stack(masks)  # [B, W]
    return imgs, masks, paths


def build_shards(crop_dir: str, out_dir: str, shard_size: int = 5000) -> None:
    """
    Encode crops into embeddings and save them as shard .pt files.

    Args:
        crop_dir: Directory containing cropped images.
        out_dir: Output directory for embedding shards.
        shard_size: Number of samples per shard file.
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = CropDataset(crop_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)

    model = CNNEncoder(d_model=256).eval()
    all_records, shard_idx = [], 0

    with torch.no_grad():
        for imgs, masks, paths in loader:
            embs = model(imgs)  # [B, T, D]
            for e, m, p in zip(embs, masks, paths):
                all_records.append({"path": p, "emb": e.half().cpu(), "mask": m.cpu()})
            if len(all_records) >= shard_size:
                out_path = os.path.join(out_dir, f"embeds_{shard_idx:05d}.pt")
                torch.save(all_records, out_path)
                print("Saved:", out_path)
                all_records, shard_idx = [], shard_idx + 1

    if all_records:
        out_path = os.path.join(out_dir, f"embeds_{shard_idx:05d}.pt")
        torch.save(all_records, out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    crop_dir = "cropped"
    out_dir = "embedding/shards"
    build_shards(crop_dir, out_dir, shard_size=5000)
