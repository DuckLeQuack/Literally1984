# train_resnet.py (beast mode edition)

import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Config
DATA_DIR = "data/"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = len(os.listdir(DATA_DIR))
ALPHA = 0.4  # mixup hyperparameter

# Transforms (no random resize crop or rotation due to fixed camera)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset split
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
import json
with open("class_names.json", "w") as f:
    json.dump(full_dataset.dataset.classes, f)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Model with dropout
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(device)

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Cosine Annealing LR Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 🔀 Mixup Function
def mixup_data(x, y, alpha=ALPHA):
    if alpha <= 0:
        return x, y, y, 1
    lam = random.betavariate(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_loss(criterion, preds, y_a, y_b, lam):
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

# Best val tracking
best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_correct = train_total = 0
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 🔀 Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
        outputs = model(inputs)
        loss = mixup_loss(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        # Estimate accuracy based on dominant label
        train_correct += (preds == targets_a).sum().item() * lam + (preds == targets_b).sum().item() * (1 - lam)
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step()  # update learning rate

    print(f"[{epoch+1}/{EPOCHS}] 🧠 Train Acc: {train_acc:.4f} | 🧪 Val Acc: {val_acc:.4f} | Loss: {train_loss:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnet_beast_best.pth")
        print(f"💾 Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}")

print("✅ Training complete.")
#test2