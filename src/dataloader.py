import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2


class DataloaderProj(Dataset):
    def __init__(
        self, image_paths, labels, tokenizer, img_size=(64, 256), type="train"
    ):
        """
        image_paths : list of image file paths
        labels      : list of corresponding text labels
        tokenizer   : instance of Tokenizer class (SentencePiece)
        img_size    : resize images to this size (H,W)
        type        : it determine the type of transform (Aug)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.img_size = img_size
        if type == "train":
            self.transform = v2.Compose(
                [
                    # New version of V2 API
                    v2.ToImage(),
                    v2.Resize(img_size),
                    v2.ColorJitter(0.1, 0.1),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(img_size),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        # Encode label with SentencePiece tokenizer
        label_text = self.labels[idx]
        token_ids = torch.tensor(self.tokenizer.encode(label_text), dtype=torch.long)

        return img, token_ids
