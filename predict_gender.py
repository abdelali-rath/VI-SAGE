# how to use:
# python predict_gender.py path/zum/bild.jpg

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

CHECKPOINT = "checkpoints/utk_gender_mobilenet.pt"

# MODEL DEFINITION (same as training)
class GenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Identity()
        self.classifier = nn.Linear(in_features, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


# TRANSFORM FOR INFERENCE
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# PREDICTION FUNCTION
def predict_gender(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = GenderNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    # Mapping
    label = "male" if pred.item() == 0 else "female"
    confidence = round(conf.item(), 4)

    return label, confidence


# CLI USAGE
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_gender.py <image_path>")
    else:
        img_path = sys.argv[1]
        label, conf = predict_gender(img_path)
        print(f"Prediction: {label} (confidence: {conf})")
