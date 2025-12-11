import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class SimpleGenderModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.fc.in_features, 2)  # male/female
        backbone.fc = nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.fc(feats)
        return out


class GenderPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        backbone = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
        self.model = SimpleGenderModel(backbone)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        self.labels = ["female", "male"]

    def predict_from_pil(self, img: Image.Image):
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        return {
            "gender": self.labels[pred.item()],
            "confidence": float(conf.item())
        }
