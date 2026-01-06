import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utk_loader import UTKFaceMultiTask
from models import MultiTaskModel


DATA_PATH = r"C:\Users\meist\Downloads\UTKFace"
CHECKPOINT_ETHNICITY = "checkpoints/ethnicity_model.pt"

BATCH_SIZE = 64
AGE_TOLERANCE = 5  # Abweichung von +- 5 Jahren gilt noch als "Richtig"

# Schriftgrößen
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def visualize_overall_performance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calculating on: {device}")

    # Daten laden
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(DATA_PATH):
        print(f"FEHLER: Pfad nicht gefunden: {DATA_PATH}")
        return

    dataset = UTKFaceMultiTask(root=DATA_PATH, transform=transform)
    
    # 20% der Daten für schnelleren Durchlauf
    val_size = int(0.2 * len(dataset)) 
    _, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modell laden
    print("Lade Modell...")
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
    
    if os.path.exists(CHECKPOINT_ETHNICITY):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_ETHNICITY, map_location=device))
        except:
             model.load_state_dict(torch.load(CHECKPOINT_ETHNICITY, map_location=device), strict=False)
    
    model.to(device)
    model.eval()

    # Zähler für Richtigkeit
    total_count = 0
    correct_gender = 0
    correct_ethnicity = 0
    correct_age = 0
    correct_all_combined = 0  # Perfect Match

    print(f"Teste {len(val_ds)} Bilder...")

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            
            # Prüfen ob alle Targets da sind
            if 'age' not in targets or 'gender' not in targets or 'ethnicity' not in targets:
                continue

            # Vorhersage
            outputs = model(images)
            
            # Predictions
            # Age
            if 'age' in outputs:
                pred_age = outputs['age']
            elif 'age_output' in outputs:
                pred_age = outputs['age_output']
            
            # Gender
            gender_probs = torch.softmax(outputs['gender_logits'], dim=1)
            _, pred_gender = torch.max(gender_probs, 1)
            
            # Ethnicity
            _, pred_eth = torch.max(outputs['ethnicity_logits'], 1)

            # TARGETS
            true_age = targets['age'].to(device)
            true_gender = targets['gender'].to(device)
            true_eth = targets['ethnicity'].to(device)

            # VERGLEICH (Batch-weise)
            # Gender Richtig?
            gender_matches = (pred_gender == true_gender)
            
            # Ethnicity Richtig?
            eth_matches = (pred_eth == true_eth)
            
            # Alter Richtig ? (Innerhalb Toleranz)
            age_diff = torch.abs(pred_age - true_age)
            age_matches = (age_diff <= AGE_TOLERANCE)

            # Alles Richtig ? (Logisches UND)
            all_matches = gender_matches & eth_matches & age_matches

            # Zählen
            total_count += images.size(0)
            correct_gender += gender_matches.sum().item()
            correct_ethnicity += eth_matches.sum().item()
            correct_age += age_matches.sum().item()
            correct_all_combined += all_matches.sum().item()

    if total_count == 0:
        print("Keine validen Daten gefunden.")
        return

    # Prozentwerte berechnen
    acc_gender = (correct_gender / total_count) * 100
    acc_eth = (correct_ethnicity / total_count) * 100
    acc_age = (correct_age / total_count) * 100
    acc_combined = (correct_all_combined / total_count) * 100

    print("Generiere Overall-Diagramm...")
    
    # Daten für Plot
    categories = ['Gender', 'Ethnicity', f'Age\n(±{AGE_TOLERANCE} Years)', 'PERFECT\nMATCH\n(All 3 Correct)']
    values = [acc_gender, acc_eth, acc_age, acc_combined]
    colors = ['#3498DB', '#9B59B6', '#F39C12', '#27AE60']

    plt.figure(figsize=(10, 7))
    
    bars = plt.bar(categories, values, color=colors, edgecolor='black', alpha=0.9, width=0.6)
    
    plt.title("Total System Accuracy (Multi-Task)", fontweight='bold', pad=20)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Werte über Balken
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig('overall_performance.png', dpi=300)
    print(f"-> Gespeichert: 'overall_performance.png'")
    print(f"   (Gender: {acc_gender:.1f}%, Eth: {acc_eth:.1f}%, Age: {acc_age:.1f}%, Combined: {acc_combined:.1f}%)")
    plt.show()

if __name__ == "__main__":
    visualize_overall_performance()