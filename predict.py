import torch
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import json

# Settings
MODEL_PATH = "resnet_beast_best.pth"
IMAGE_PATH = "data/4015/4015-27.png"  # 👈 change this to test other images
DATA_DIR = "data/"  # same as training dir for raw (non-preprocessed) image data

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load label names from JSON file
with open("class_names.json") as f:
    class_names = json.load(f)

# Load image
def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# Load model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, len(class_names))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Predict
img = load_image(IMAGE_PATH)
with torch.no_grad():
    output = model(img)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_idxs = probabilities.topk(3, dim=1)

print("🔮 Top 3 Predictions:")
for i in range(3):
    label = class_names[top_idxs[0][i].item()]
    confidence = top_probs[0][i].item() * 100
    print(f"{i+1}: {label} ({confidence:.2f}%)")
    #tester fewfe