from CNNEncoder import CNNEncoderVisualize, visualize_feature_maps
from crop import prepare_crops_for_cnn
import torch

# 1️⃣ Prepare crops
image_path = "doc.jpg"
crops = prepare_crops_for_cnn(image_path)  # [num_crops, 3, H, W]
if crops.shape[0] == 0:
    print("No crops detected.")
else:
    print("Crops shape:", crops.shape)

# 2️⃣ Initialize CNN encoder for visualization
encoder_viz = CNNEncoderVisualize(d_model=256, in_channels=3).eval()

# 3️⃣ Forward pass to get embeddings + intermediate feature maps
with torch.no_grad():
    embeddings, x1, x2, x3 = encoder_viz(crops)

print("Embeddings shape:", embeddings.shape)
print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)
print("x3 shape:", x3.shape)

# 4️⃣ Visualize feature maps of the first crop
print("Visualizing first conv block (x1)")
visualize_feature_maps(x1, num_channels=3)

print("Visualizing second conv block (x2)")
visualize_feature_maps(x2, num_channels=3)

print("Visualizing third conv block (x3)")
visualize_feature_maps(x3, num_channels=3)