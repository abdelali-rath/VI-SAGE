# src/gender_infer.py

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
from PIL import Image
import os


class GenderInference:
    """
    Uses MobileNetV3-Large trained for gender classification.
    Perfectly matches checkpoint keys that start with 'backbone.'.
    """
    def __init__(self, checkpoint_path="checkpoints/utk_gender_mobilenet.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load pretrained MobileNetV3 Large
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        # Replace classifier head (653 -> 2 classes)
        in_features = base.classifier[3].in_features
        base.classifier[3] = nn.Linear(in_features, 2)

        self.model = base

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=self.device)

        # Fix key mismatch: backbone.* → nothing
        new_state = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                new_state[k[len("backbone."):]] = v
            else:
                new_state[k] = v

        self.model.load_state_dict(new_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.classes = ["female", "male"]

    def predict(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)
        return self.classes[idx.item()], float(conf.item())
