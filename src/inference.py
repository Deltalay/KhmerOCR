# infer.py
import torch
from PIL import Image
from torchvision import transforms

from model import PretrainedOCR
from tokenizer import Tokenizer

device = "cuda"

def ctc_greedy_decode(logits, blank_id):
    max_ids = logits.argmax(dim=2)          # (T, B)
    max_ids = max_ids.permute(1, 0)         # (B, T)
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

# config
checkpoint_path = r"C:\Users\b2324\Desktop\KhmerOCR\checkpoints\epoch_23.pt"
image_paths = [r"C:\Users\b2324\Desktop\KhmerOCR\image\test2.png"]  # replace with your images

# init
tokenizer = Tokenizer()
vocab_size = len(tokenizer)
model = PretrainedOCR(num_classes=vocab_size).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
])

# run
with torch.no_grad():
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)          # (1,3,224,224)
        logits = model(x).permute(1, 0, 2)                  # (T, B, C)
        pred_ids = ctc_greedy_decode(logits, tokenizer.blank_id())[0]
        text = tokenizer.decode(pred_ids)
        print(p, "=>", text)
