import os
import torch
from torch.utils.data import DataLoader

# collate function pads sequences
def collate(records):
    embs = [r["emb"] for r in records]
    masks = [r["mask"] for r in records]
    paths = [r["path"] for r in records]
    embs_padded = torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)   # [B,Tmax,D]
    masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True) # [B,Tmax]
    return embs_padded, masks_padded, paths

# dataset that streams from shard .pt files
class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, shard_paths):
        self.samples = []
        for sp in shard_paths:
            data = torch.load(sp, map_location="cpu")
            self.samples.extend(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Example usage
if __name__ == "__main__":
    shard_dir = r"C:\Users\b2324\Desktop\KhmerOCR\embedding\shards"
    shard_paths = [os.path.join(shard_dir,f) for f in os.listdir(shard_dir) if f.endswith(".pt")]

    dataset = ShardDataset(shard_paths)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate, shuffle=False)

    for embs, masks, paths in loader:
        print("Batch embeddings:", embs.shape)  # [B,Tmax,D]
        print("Batch masks:", masks.shape)      # [B,Tmax]
        # forward into your ViT decoder here:
        # logits = vit_decoder(embs, masks)
        break
