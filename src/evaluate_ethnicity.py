import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


from utk_loader import UTKFaceMultiTask
from models import MultiTaskModel

# ------------- CONFIGURATION -------------

DATA_PATH = r"C:\Users\meist\Downloads\UTKFace"
CHECKPOINT_PATH = "checkpoints/ethnicity_model.pt"
OUTPUT_DIR = "eval_results"
BATCH_SIZE = 64

ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Other"]

def evaluate_with_visuals():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device}")

    # 1. Load Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = UTKFaceMultiTask(root=DATA_PATH, transform=transform)
    
    # Validation split logic
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    print("Loading model...")
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Run Inference and Collect Confidence
    all_preds = []
    all_labels = []
    all_confs = [] # Confidence scores
    
    print("Running inference (collecting confidence scores)...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            labels = targets["ethnicity"].to(device)

            outputs = model(images)
            
            # Calculate Probabilities (Softmax) to get Confidence
            logits = outputs["ethnicity_logits"]
            probs = torch.softmax(logits, dim=1)
            
            # Get max probability (the confidence) and the predicted class
            confs, predicted = torch.max(probs, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    # ---VISUAL 1: CONFUSION MATRIX HEATMAP ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ETHNICITY_LABELS, 
                yticklabels=ETHNICITY_LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Ethnicity Confusion Matrix (Raw Counts)')
    
    save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"-> Saved: {save_path}")
    plt.close()

    # --- VISUAL 2: PERFORMANCE BAR CHART ---
    print("Generating Metrics Chart...")
    report = classification_report(all_labels, all_preds, target_names=ETHNICITY_LABELS, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    # Filter only the classes (remove accuracy/macro avg rows)
    class_metrics = df.iloc[:5] 
    
    class_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Metrics by Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'metrics_chart.png')
    plt.savefig(save_path)
    print(f"-> Saved: {save_path}")
    plt.close()

    # --- VISUAL 3: CONFIDENCE DISTRIBUTION (REPLACEMENT) ---
    print("Generating Confidence Boxplot...")
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Ethnicity': [ETHNICITY_LABELS[i] for i in all_preds],
        'Confidence': all_confs
    })

    plt.figure(figsize=(10, 6))
    # Boxplot shows median, quartiles, and outliers
    sns.boxplot(x='Ethnicity', y='Confidence', data=plot_data, order=ETHNICITY_LABELS, palette="Set2")
    
    plt.title('Model Confidence Distribution per Predicted Ethnicity')
    plt.ylabel('Confidence Score (0.0 - 1.0)')
    plt.xlabel('Predicted Ethnicity')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add a red line at 0.5 (random guess threshold for binary, but useful visual anchor)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Low Confidence Threshold')
    plt.legend(loc='lower right')
    
    save_path = os.path.join(OUTPUT_DIR, 'confidence_boxplot.png')
    plt.savefig(save_path)
    print(f"-> Saved: {save_path}")
    plt.close()
    
    print(f"\nDone! Visualizations saved to: {os.path.abspath(OUTPUT_DIR)}")



if __name__ == "__main__":
    evaluate_with_visuals()