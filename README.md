# KhmerOCR

KhmerOCR is an end-to-end Optical Character Recognition (OCR) system for Khmer text. The pipeline leverages **YOLO** for text detection, crops detected regions, and then processes them with a hybrid **CNN + Vision Transformer (ViT)** architecture for recognition.

This repository emphasizes **readability, modularity, and maintainability**, inspired by Linux kernel coding principles.

---

## Table of Contents

- [KhmerOCR](#khmerocr)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Pipeline](#pipeline)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Run Detection + Recognition](#run-detection--recognition)
    - [Train Recognition Model](#train-recognition-model)
    - [Evaluate Model](#evaluate-model)
  - [Contributing](#contributing)
  - [License](#license)

---

## Project Overview

KhmerOCR is designed to detect and recognize Khmer text in images efficiently. The system is structured to be modular:

1. **Text detection**: YOLO detects text bounding boxes in an image.
2. **Cropping**: Detected regions are cropped to feed into the recognition network.
3. **Recognition**: A hybrid **CNN + ViT** model predicts the text in each cropped region.

This design allows for **flexible replacement** of components while maintaining a clean and readable codebase.

---

## Pipeline

```

Input Image → YOLO Detector → Crop Detected Text → CNN Encoder → ViT Decoder → Text Output

```

1. **YOLO Detector**
   - Detects text bounding boxes in the input image.
   - Supports fast real-time detection.

2. **Cropping Module**
   - Crops each detected bounding box from the image.
   - Ensures uniform input size for the recognition network.

3. **CNN + ViT Recognition**
   - **CNN** extracts spatial features from cropped text regions.
   - **ViT** processes features sequentially for character-level recognition.
   - Combines local and global context for high accuracy.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Deltalay/KhmerOCR
cd KhmerOCR
````

2. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Detection + Recognition

```bash
python main.py --image_path path/to/image.jpg --mode infer
```

### Train Recognition Model

```bash
python training/train.py --config configs/default.yaml
```

### Evaluate Model

```bash
python training/evaluate.py --checkpoint path/to/model.pth
```

---

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding principles, folder guidelines, and contribution workflow.

---

## License

This project is licensed under the MIT License.

