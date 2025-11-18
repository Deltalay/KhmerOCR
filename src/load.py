import os
import random
from sklearn.model_selection import train_test_split

def load_labels(label_file, base_dir="", check_exists=True):
    image_paths = []
    labels = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue  # skip malformed lines

            img_path, label = parts
            full_img_path = os.path.join(base_dir, img_path.replace("\\", "/"))

            if check_exists and not os.path.exists(full_img_path):
                print(f"[Warning] Missing image: {full_img_path}")
                continue

            image_paths.append(full_img_path)
            labels.append(label)

    return image_paths, labels


def split_dataset(image_paths, labels, val_ratio=0.1, seed=42):
    train_img, val_img, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_ratio, random_state=seed, shuffle=True
    )
    return (train_img, train_labels), (val_img, val_labels)
