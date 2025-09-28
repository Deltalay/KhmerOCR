import os
import torch
from torch.utils.data import DataLoader, Dataset


def collate(records: list):
    """
    Collate function for batching shard records.

    Pads embeddings and masks to the maximum sequence length in the batch.

    Args:
        records: List of dictionaries with keys "emb", "mask", "path".

    Returns:
        embs_padded: Tensor of shape [B, Tmax, D].
        masks_padded: Tensor of shape [B, Tmax].
        paths: List of file paths corresponding to the samples.
    """
    embs = [r["emb"] for r in records]
    masks = [r["mask"] for r in records]
    paths = [r["path"] for r in records]
    embs_padded = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)   # [B, Tmax, D]
    masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True) # [B, Tmax]
    return embs_padded, masks_padded, paths


class ShardDataset(Dataset):
    """Dataset that streams records from saved shard .pt files."""

    def __init__(self, shard_paths: list[str]) -> None:
        self.samples = []
        for sp in shard_paths:
            data = torch.load(sp, map_location="cpu")
            self.samples.extend(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


if __name__ == "__main__":
    shard_dir = "embedding/shards"
    shard_paths = [os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".pt")]

    dataset = ShardDataset(shard_paths)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate, shuffle=False)

    for embs, masks, paths in loader:
        print("Batch embeddings:", embs.shape)  # [B, Tmax, D]
        print("Batch masks:", masks.shape)      # [B, Tmax]
        # forward into your ViT decoder here:
        # logits = vit_decoder(embs, masks)
        break
