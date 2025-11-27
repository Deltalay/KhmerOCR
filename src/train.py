import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.transforms import v2
from ViTTest import KhmerOCRViT
from dataloader import DataloaderProj
from load import load_labels, split_dataset
from torchmetrics.text import CharErrorRate, WordErrorRate
from tokenizer import Tokenizer

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = Tokenizer()
PAD_ID = tokenizer.get_size()  # New ID for padding
vocab_size = tokenizer.get_size() + 1  # +1 for PAD token

# -------------------------------
# Collate function for CE
# -------------------------------
def collate_fn_ce(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)  # (B, C, H, W)
    
    max_len = max(len(l) for l in labels)
    padded_labels = torch.full((len(labels), max_len), fill_value=PAD_ID, dtype=torch.long)
    
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)
    
    return images, padded_labels

# -------------------------------
# Load datasets
# -------------------------------
image_paths, labels = load_labels("labels.txt")
(train_img, train_labels), (val_img, val_labels) = split_dataset(image_paths, labels, val_ratio=0.2)

train_dataset = DataloaderProj(train_img, train_labels, tokenizer, img_size=(64, 256))
val_dataset = DataloaderProj(val_img, val_labels, tokenizer, img_size=(64, 256), type="validation")

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn_ce)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn_ce)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = KhmerOCRViT(num_classes=vocab_size, embed_dim=512, num_heads=16).to(device)
ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# -------------------------------
# Metrics
# -------------------------------
cer_metric = CharErrorRate().to(device)
wer_metric = WordErrorRate().to(device)

# -------------------------------
# Training
# -------------------------------
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
num_epochs = 100

for epoch in range(num_epochs):
    cer_metric.reset()
    wer_metric.reset()

    # -------------------
    # TRAINING
    # -------------------
    model.train()
    train_loss = 0.0

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)  # (B, T_target)

        B = images.size(0)
        outputs = model(images)  # (B, T_out, V)
        T_out = outputs.size(1)
        vocab_size = outputs.size(2)

        # Pad/truncate targets to match model output length
        padded_targets = torch.full((B, T_out), fill_value=PAD_ID, dtype=torch.long, device=targets.device)
        for i, tgt_seq in enumerate(targets):
            length = min(len(tgt_seq), T_out)
            padded_targets[i, :length] = tgt_seq[:length]

        # Compute CE loss
        loss = ce_loss(outputs.reshape(-1, vocab_size), padded_targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (step + 1) % 100 == 0:
            print(f"[Train {step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # -------------------
    # VALIDATION
    # -------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            B = images.size(0)
            outputs = model(images)  # (B, T_out, V)
            T_out = outputs.size(1)
            vocab_size = outputs.size(2)

            # Pad/truncate targets to match model output length
            padded_targets = torch.full((B, T_out), fill_value=PAD_ID, dtype=torch.long, device=targets.device)
            for i, tgt_seq in enumerate(targets):
                length = min(len(tgt_seq), T_out)
                padded_targets[i, :length] = tgt_seq[:length]

            # Compute CE loss
            loss = ce_loss(outputs.reshape(-1, vocab_size), padded_targets.reshape(-1))
            val_loss += loss.item()

            # -------------------
            # Token-level CER/WER
            # -------------------
            pred_tokens = outputs.argmax(dim=2)  # (B, T_out)
            pred_texts = []
            target_texts = []

            for pred_seq, tgt_seq in zip(pred_tokens, padded_targets):
                pred_seq = [t.item() for t in pred_seq if t.item() != PAD_ID]
                tgt_seq = [t.item() for t in tgt_seq if t.item() != PAD_ID]
                pred_texts.append(tokenizer.decode(pred_seq))
                target_texts.append(tokenizer.decode(tgt_seq))

            cer_metric.update(pred_texts, target_texts)
            wer_metric.update(pred_texts, target_texts)

    avg_val_loss = val_loss / len(val_loader)
    cer = cer_metric.compute().item()
    wer = wer_metric.compute().item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"CER: {cer:.4f} | WER: {wer:.4f}"
    )

    # -------------------
    # Save checkpoint
    # -------------------
    checkpoint_path = f"./checkpoints/epoch_{epoch+1}.pt"
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")