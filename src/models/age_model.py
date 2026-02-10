import torch
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ---------------------------------------------------------
# MobileNetV3 Age Regression Model
# ---------------------------------------------------------
class AgeModel(nn.Module):
    """
    A neural network model for age regression using MobileNetV3 backbone.
    
    This model takes an image as input and predicts the age as a continuous value.
    """
    def __init__(self, pretrained=True):
        """
        Initialize the AgeModel.
        
        Args:
            pretrained (bool): Whether to use pretrained weights for MobileNetV3.
        """
        super().__init__()

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W].
            
        Returns:
            torch.Tensor: Predicted age values.
        """
        return self.backbone(x)


# ---------------------------------------------------------
# Age Inference Wrapper
# ---------------------------------------------------------
class AgeInference:
    """
    Wrapper class for age prediction inference using a trained AgeModel.
    
    Handles model loading, preprocessing, and prediction with age denormalization.
    """
    def __init__(self, checkpoint_path, device="cpu", debug=True):
        """
        Initialize the AgeInference wrapper.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint file.
            device (str): Device to run the model on ('cpu' or 'cuda').
            debug (bool): Whether to print debug information.
        """
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
        """
        Preprocess a PIL image for model input.
        
        Resizes the image to 224x224, normalizes using ImageNet statistics,
        and converts to a tensor on the appropriate device.
        
        Args:
            pil_img (PIL.Image): Input PIL image.
            
        Returns:
            torch.Tensor: Preprocessed tensor of shape [1, 3, 224, 224].
        """
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
        """
        Predict age from a preprocessed tensor.
        
        Performs inference, denormalizes the output, and categorizes the age.
        
        Args:
            tensor (torch.Tensor): Preprocessed input tensor.
            
        Returns:
            dict: Dictionary containing 'age', 'age_range', and 'confidence'.
        """
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
            age_range = "Child"
        elif age <= 19:
            age_range = "Teen"
        elif age <= 29:
            age_range = "Young adult"
        elif age <= 44:
            age_range = "Adult"
        elif age <= 59:
            age_range = "Middle age"
        else:
            age_range = "Senior"

        return {
            "age": age,
            "age_range": age_range,
            "confidence": None
        }


    def predict_from_pil(self, pil_img):
        """
        Convenience method to predict age directly from a PIL image.
        
        Args:
            pil_img (PIL.Image): Input PIL image.
            
        Returns:
            dict: Dictionary containing 'age', 'age_range', and 'confidence'.
        """
        tensor = self.preprocess(pil_img)
        return self.predict(tensor)