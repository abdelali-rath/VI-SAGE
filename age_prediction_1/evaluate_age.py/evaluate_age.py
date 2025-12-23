import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = r"C:\Users\ijaha\Downloads\UTKFACE"
CHECKPOINT = "checkpoints/utk_age_mobilenet.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TRANSFORM (IDENTISCH ZUM TRAINING)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# MODEL (IDENTISCH ZUM TRAINING)
# =========================
class AgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.regressor(x).squeeze(1)

# =========================
# LOAD MODEL
# =========================
model = AgeNet().to(device)
state_dict = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded successfully")

# =========================
# EVALUATION
# =========================
y_true = []
y_pred = []

print("Evaluating age model...")

for file in tqdm(os.listdir(DATA_DIR)):
    if not file.lower().endswith(".jpg"):
        continue

    try:
        age = int(file.split("_")[0])
    except:
        continue

    img_path = os.path.join(DATA_DIR, file)
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).item()

    y_true.append(age)
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# METRICS
# =========================
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\n==============================")
print("AGE REGRESSION REPORT")
print("==============================")
print(f"Samples : {len(y_true)}")
print(f"MAE     : {mae:.2f} years")
print(f"RMSE    : {rmse:.2f} years")

# =========================
# PLOTS
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, s=5, alpha=0.3)
plt.plot([0, 100], [0, 100], "r--")
plt.xlabel("True Age")
plt.ylabel("Predicted Age")
plt.title("Age Prediction – True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

errors = y_pred - y_true
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50)
plt.xlabel("Prediction Error (years)")
plt.ylabel("Count")
plt.title("Age Prediction Error Distribution")
plt.grid(True)
plt.tight_layout()
plt.show()
