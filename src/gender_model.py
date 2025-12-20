import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class GenderNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load MobileNetV3-Large backbone
        self.backbone = mobilenet_v3_large(weights=None)

        # Remove classification head – we will replace it exactly
        in_channels = self.backbone.classifier[0].in_features  # 960
        hidden_dim = self.backbone.classifier[0].out_features  # 1280

        # Rebuild classifier EXACTLY like checkpoint
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Hardswish(),
        )

        # Your gender head (matches checkpoint: 2 classes)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)  # 960 -> 1280
        x = self.classifier(x)           # 1280 -> 2
        return x


class GenderInference:
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = torch.device(device)
        self.model = GenderNet().to(self.device)

        # Load checkpoint exactly
        state = torch.load(checkpoint_path, map_location=self.device)

        # rename nothing -> matches 100%
        self.model.load_state_dict(state, strict=True)

        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.classes = ["male", "female"]

        print("Gender checkpoint loaded successfully.")

    @torch.no_grad()
    def predict(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        logits = self.model(img_tensor)
        probs = self.softmax(logits)[0]
        idx = probs.argmax().item()
        return {
            "gender": self.classes[idx],
            "confidence": float(probs[idx]),
        }
