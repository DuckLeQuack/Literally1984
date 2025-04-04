import torch
import clip
from PIL import Image

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define class labels (same as your dataset folders)
class_labels = [
    "4011",
    "4015",
    "4088",
    "4196",
    "7020097009819",
    "7020097026113",
    "7023026089401",
    "7035620058776",
    "7037203626563",
    "7037206100022",
    "7038010009457",
    "7038010013966",
    "7038010021145",
    "7038010054488",
    "7038010068980",
    "7039610000318",
    "7040513000022",
    "7040513001753",
    "7040913336684",
    "7044610874661",
    "7048840205868",
    "7071688004713",
    "7622210410337",
    "90433917",
    "90433924",
    "94011"
]


# Load and preprocess test image
image_path = "test.png"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Convert text labels into CLIP-readable format
text_inputs = clip.tokenize([f"an image of {label}" for label in class_labels]).to(device)

# Get predictions
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)
    logits_per_image, logits_per_text = model(image, text_inputs)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# Print prediction
predicted_class = class_labels[probs.argmax()]
print(f"Predicted class: {predicted_class}")
