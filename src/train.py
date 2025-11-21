import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import DataloaderProj
from load import load_labels, split_dataset
from model import PretrainedOCR

device = "cuda"
from tokenizer import Tokenizer

tokenizer = Tokenizer()


def collate_fn_ctc(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths


image_paths, labels = load_labels("labels.txt")
(train_img, train_labels), (val_img, val_labels) = split_dataset(
    image_paths, labels, val_ratio=0.2
)

train_dataset = DataloaderProj(train_img, train_labels, tokenizer, img_size=(224, 224))
val_dataset = DataloaderProj(val_img, val_labels, tokenizer, img_size=(224, 224))

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_ctc
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_ctc
)

vocab_size = tokenizer.get_size()
model = PretrainedOCR(num_classes=vocab_size).to(device)

ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id(), zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    i = 0
    train_loss = 0
    for images, targets, target_lengths in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)
        input_lengths = torch.full(
            size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long
        ).to(device)

        loss = ctc_loss(outputs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f"[Step {i + 1}/{len(train_loader)}] Training Loss: {loss.item():.4f}"
            )
        i = i + 1
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long
            ).to(device)

            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    )

    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
