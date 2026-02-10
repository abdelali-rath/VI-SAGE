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
    def __init__(self, checkpoint_path, device="cpu", debug=False):
        self.device = torch.device(device)
        self.debug = debug
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

        # Load weights with non-strict to allow minor head differences
        try:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Warning: Gender model failed to load (strict): {e}")
            self.model.load_state_dict(state, strict=True)
            missing, unexpected = [], []

        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        
        # CRITICAL FIX: Check actual model behavior to determine correct label mapping
        # Standard UTKFace convention: index 0 = male, index 1 = female
        # BUT your probs show [0.994, 0.005] predicting "male", which is CORRECT for index 0
        # So the mapping should be:
        self.classes = ["male", "female"]  # Index 0=male, 1=female

        # Print concise status
        if missing:
            print(f"⚠️ Gender checkpoint loaded with {len(missing)} missing, {len(unexpected)} unexpected keys")
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

    @torch.no_grad()
    def predict(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        logits = self.model(img_tensor)
        probs = self.softmax(logits)[0]
        
        # Get the predicted class and confidence
        idx = probs.argmax().item()
        confidence = float(probs[idx])
        gender_label = self.classes[idx]
        
        if self.debug:
            print(f"[GENDER DEBUG] probs={probs.cpu().numpy()}, predicted={gender_label}, conf={confidence:.4f}, idx={idx}")
        
        return {
            "gender": gender_label,
            "confidence": confidence,
        }