import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms


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

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=self.device)

        # Unwrap common checkpoint containers
        if isinstance(state, dict):
            for k in ("model_state", "model_state_dict", "state_dict"):
                if k in state:
                    state = state[k]
                    break

        # Detect whether checkpoint keys belong to a bare MobileNet (e.g. 'features.*')
        # or are already prefixed (e.g. 'backbone.features.*'). Remap as needed.
        has_backbone_keys = any(k.startswith('backbone.') for k in state.keys()) if isinstance(state, dict) else False
        has_features_keys = any(k.startswith('features.') for k in state.keys()) if isinstance(state, dict) else False

        if has_features_keys and not has_backbone_keys:
            print("⚠️ Remapping gender checkpoint keys (features.* -> backbone.features.*)")
            new_state = {}
            for k, v in state.items():
                if k.startswith('features.') or k.startswith('classifier.'):
                    new_state[f'backbone.{k}'] = v
                else:
                    new_state[k] = v
            state = new_state

        # Load weights with non-strict to allow minor head differences and report mismatches
        try:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
        except Exception as e:
            # Fall back to strict load to surface errors
            print(f"Warning: Gender model failed to load (strict): {e}")
            self.model.load_state_dict(state, strict=True)
            missing, unexpected = [], []

        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        # Checkpoint mapping: index 0 -> male, 1 -> female
        self.classes = ["male", "female"]

        # Print concise status
        if missing:
            print(f"Warning: Gender checkpoint loaded with missing keys: {len(missing)} missing, {len(unexpected)} unexpected")
        else:
            print("✅ Gender checkpoint loaded successfully.")

        # Preprocessing (convenience)
        self._pil_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_from_pil(self, pil_img):
        """Convenience helper: accept PIL Image and return same dict as `predict`."""
        x = self._pil_transform(pil_img).unsqueeze(0).to(self.device)
        return self.predict(x)

        # Signal successful load to console (appears during demo startup)
    # end predict_from_pil

        
    # Note: loading confirmation printed in constructor

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
