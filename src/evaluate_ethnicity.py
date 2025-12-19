import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

from utk_loader import UTKFaceMultiTask
from models import MultiTaskModel

# CONFIG
DATA_PATH = r"C:\Users\meist\Downloads\UTKFace"
CHECKPOINT_PATH = "checkpoints/ethnicity_model.pt"
BATCH_SIZE = 64

# Labels matching UTKFace and your infer.py
ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Other"]

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}...")

    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = UTKFaceMultiTask(root=DATA_PATH, transform=transform)
    
    # Use the same split logic to ensure we test on unseen data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Load Model
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}. Train first!")
        return

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Run Inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            labels = targets["ethnicity"].to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs["ethnicity_logits"], 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Print Statistics
    print("\n" + "="*40)
    print("       ETHNICITY CLASSIFICATION REPORT")
    print("="*40)
    
    # Per-class accuracy
    print(classification_report(all_labels, all_preds, target_names=ETHNICITY_LABELS, digits=3))

    # Confusion Matrix
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate()