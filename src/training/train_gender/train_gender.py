"""
Training script for gender classification on the UTKFace dataset using
MobileNetV3-Large as a fast backbone.

Usage (from project root):
    python src/training/train_gender/train_gender.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from tqdm.auto import tqdm

try:
    # When run as a module: python -m src.training.train_gender.train_gender_utk
    from .utk_loader import UTKFaceGender
except ImportError:  # pragma: no cover - fallback for direct script execution
    # When run directly from the project root: python src/training/train_gender/train_gender.py
    from utk_loader import UTKFaceGender

# CONFIG
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "UTKFace")
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "utk_gender_model.pt")

# DATASET
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

dataset = UTKFaceGender(DATA_PATH, transform=transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# MODEL (MobileNetV3-Large — VERY FAST)
class GenderNet(nn.Module):
    def __init__(self):
        """
        Gender classifier based on MobileNetV3‑Large.

        Removes the original classification head and replaces it with a
        linear layer producing two logits (male / female).
        """
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()  # remove original head

        self.classifier = nn.Linear(in_features, 2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Batch of images of shape [B, 3, H, W].

        Returns:
            Tensor of shape [B, 2] with logits for (male, female).
        """
        feat = self.backbone(x)
        return self.classifier(feat)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = GenderNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# TRAINING LOOP
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": loss.item()})

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

# SAVE MODEL
os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
torch.save(model.state_dict(), CHECKPOINT)
print(f"Model saved → {CHECKPOINT}")
