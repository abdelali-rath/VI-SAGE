import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from utk_loader import UTKFaceMultiTask 
from models import MultiTaskModel


# ---------------- CONFIGURATION ----------------
DATA_PATH = r"C:\Users\meist\Downloads\UTKFace"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "ethnicity_model.pt"

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4

# Device configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ---------------- DATA PREPARATION ----------------
# Standard ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Initialize Dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Directory not found: {DATA_PATH}")

# Load dataset using your modified loader
dataset = UTKFaceMultiTask(root=DATA_PATH, transform=transform)

# Split Train/Val (90% / 10%)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Windows often requires num_workers=0 or 2. If it freezes, set num_workers=0.
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------- MODEL SETUP ----------------
# Initialize model with 5 ethnicity classes (White, Black, Asian, Indian, Other)
model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ---------------- TRAINING LOOP ----------------
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    print(f"Starting training on {len(train_ds)} images...")

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN STEP ---
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        
        for images, targets in loop:
            images = images.to(DEVICE)
            # Extract ethnicity labels from the dictionary
            labels = targets["ethnicity"].to(DEVICE)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            ethnicity_logits = outputs["ethnicity_logits"]
            
            # Calculate loss ONLY on ethnicity
            loss = criterion(ethnicity_logits, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATION STEP ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                labels = targets["ethnicity"].to(DEVICE)

                outputs = model(images)
                # Get predictions
                _, predicted = torch.max(outputs["ethnicity_logits"], 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved best model to {save_path}")

if __name__ == "__main__":
    train()