import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from model import PretrainedOCR
from dataloader import DataloaderProj
from load import load_labels, split_dataset

device = "cuda"
from tokenizer import Tokenizer
from Levenshtein import distance as levenshtein_distance
import jiwer

tokenizer = Tokenizer()

def collate_fn_ctc(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths

def ctc_greedy_decode(logits, blank_id):
    max_ids = logits.argmax(dim=2)
    max_ids = max_ids.permute(1, 0)
    seqs = []
    for seq in max_ids:
        prev = None
        out = []
        for t in seq.tolist():
            if t != blank_id and t != prev:
                out.append(t)
            prev = t
        seqs.append(out)
    return seqs

image_paths, labels = load_labels("labels.txt")
(train_img, train_labels), (val_img, val_labels) = split_dataset(image_paths, labels, val_ratio=0.2)

train_dataset = DataloaderProj(train_img, train_labels, tokenizer, img_size=(224, 224))
val_dataset = DataloaderProj(val_img, val_labels, tokenizer, img_size=(224, 224))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_ctc)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_ctc)

vocab_size = len(tokenizer)
model = PretrainedOCR(num_classes=vocab_size).to(device)

ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id(), zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-4,  weight_decay=1e-4)

save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)

# moved resume block here
resume_path = r"C:\Users\b2324\Desktop\KhmerOCR\checkpoints\epoch_14.pt"  # set path if resuming
start_epoch = 0
if resume_path:
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']

num_epochs = 30
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss = 0
    i = 0
    for images, targets, target_lengths in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

        loss = ctc_loss(outputs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"[Step {i + 1}/{len(train_loader)}] Training Loss: {loss.item():.4f}")

        i += 1
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
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    total_char_edits = 0
    total_char_ref = 0
    total_word_err = 0.0
    total_word_ref = 0
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)

            pred_ids = ctc_greedy_decode(outputs, tokenizer.blank_id())
            idx = 0
            for b in range(images.size(0)):
                tgt_len = target_lengths[b].item()
                tgt_seq = targets[idx:idx+tgt_len].tolist()
                idx += tgt_len

                ref_text = tokenizer.decode(tgt_seq)
                pred_text = tokenizer.decode(pred_ids[b])

                total_char_edits += levenshtein_distance(pred_text, ref_text)
                total_char_ref += len(ref_text)

                total_word_err += jiwer.wer(ref_text, pred_text) * max(1, len(ref_text.split()))
                total_word_ref += max(1, len(ref_text.split()))

    cer = total_char_edits / total_char_ref if total_char_ref > 0 else 0.0
    wer = total_word_err / total_word_ref if total_word_ref > 0 else 0.0
    print(f"CER: {cer:.4f}, WER: {wer:.4f}")
