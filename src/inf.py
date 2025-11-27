import torch
from ViTTest import KhmerOCRViT
from tokenizer import Tokenizer
from PIL import Image
from torchvision.transforms import v2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = Tokenizer()
PAD_ID = tokenizer.get_size()  # New ID for padding
vocab_size = tokenizer.get_size() + 1 
model = KhmerOCRViT(num_classes=vocab_size, embed_dim=512, num_heads=16).to(device)
state = torch.load("test.pt", map_location=device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", trainable_params)
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)
untrain = total_params - trainable_params
print(untrain)
model.load_state_dict(state["model_state_dict"], strict=False)  
for n, p in model.named_parameters():
    print(n, p.shape)
model.eval()
img_size=(64, 256)
transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(img_size),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
def infer_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)                 # (1, seq_len, vocab_size)
        pred_ids = logits.argmax(-1)[0]
    pred_ids = [i.item() for i in pred_ids if i.item() != PAD_ID]

    # Decode
    text = tokenizer.decode(pred_ids)
    return text
result = infer_image("siemreap.png")
print(result)
