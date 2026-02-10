import torch
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ---------------------------------------------------------
# MobileNetV3 Age Regression Model
# ---------------------------------------------------------
class AgeModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------
# Age Inference Wrapper
# ---------------------------------------------------------
class AgeInference:
    def __init__(self, checkpoint_path, device="cpu", debug=True):
        self.device = torch.device(device)
        self.debug = debug

        self.model = AgeModel(pretrained=True).to(self.device)

        state = torch.load(checkpoint_path, map_location=self.device)

        # UTK regressor → classifier mapping
        if "regressor.weight" in state:
            print("⚠️ Remapping age checkpoint keys (regressor → classifier)")
            state = {
                "backbone.classifier.3.weight": state["regressor.weight"],
                "backbone.classifier.3.bias":   state["regressor.bias"],
            }

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        print("✅ Age checkpoint loaded successfully.")


    # -----------------------------------------------------
    # Preprocessing (ImageNet normalization)
    # -----------------------------------------------------
    def preprocess(self, pil_img):
        img = pil_img.resize((224, 224))
        arr = np.array(img).astype("float32") / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        try:
            param_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        return tensor.to(self.device, dtype=param_dtype)


    # -----------------------------------------------------
    # Predict age with UTK denormalization
    # -----------------------------------------------------
    def predict(self, tensor):
        with torch.no_grad():
            raw = self.model(tensor).item()

        # CRITICAL: UTKFace reverse normalization
        # During training, ages were normalized as: (age - 50) / 10
        # So we reverse it: age = raw * 10 + 50
        age = raw * 10.0 + 50.0

        if self.debug:
            print(f"[AGE DEBUG] raw={raw:.3f} → age={age:.1f}")

        # Clamp to realistic human age range
        age = max(0.0, min(100.0, age))

        # Age categories (German labels for UI)
        if age <= 12:
            age_range = "Kind"
        elif age <= 19:
            age_range = "Teen"
        elif age <= 29:
            age_range = "Junge\\/r Erwachsene\\/r"
        elif age <= 44:
            age_range = "Erwachsene\\/r"
        elif age <= 59:
            age_range = "Mittleres Alter"
        else:
            age_range = "Senior"

        return {
            "age": age,
            "age_range": age_range,
            "confidence": None
        }


    def predict_from_pil(self, pil_img):
        """Convenience method for PIL images"""
        tensor = self.preprocess(pil_img)
        return self.predict(tensor)