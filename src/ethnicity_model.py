import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def _make_layer(in_channels, out_channels, blocks, stride):
    """Helper to create ResNet layer"""
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    layers = []
    layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    
    return nn.Sequential(*layers)


class MultiTaskModel(nn.Module):
    """Multi-task model with shared backbone and separate heads"""
    def __init__(self, num_ethnicity_classes=5):
        super(MultiTaskModel, self).__init__()
        
        # Backbone as Sequential (matches checkpoint structure: backbone.0, backbone.1, etc.)
        self.backbone = nn.Sequential(
            # Initial conv (index 0)
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # BatchNorm (index 1)
            nn.BatchNorm2d(64),
            # ReLU (index 2)
            nn.ReLU(inplace=True),
            # MaxPool (index 3)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Layer1 (index 4)
            _make_layer(64, 64, 2, stride=1),
            # Layer2 (index 5)
            _make_layer(64, 128, 2, stride=2),
            # Layer3 (index 6)
            _make_layer(128, 256, 2, stride=2),
            # Layer4 (index 7)
            _make_layer(256, 512, 2, stride=2),
            # AdaptiveAvgPool (index 8)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Task-specific heads (match checkpoint dimensions)
        # Age head: 512 -> 128 -> 1 (regression)
        self.age_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Regression output
        )
        
        # Gender head: 512 -> 64 -> 2
        self.gender_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Ethnicity head: 512 -> 128 -> num_classes
        self.ethnicity_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_ethnicity_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        age_out = self.age_head(features)
        gender_out = self.gender_head(features)
        ethnicity_out = self.ethnicity_head(features)
        
        return age_out, gender_out, ethnicity_out


class EthnicityInference:
    
    # Ethnicity class labels (kurz, DE)
    ETHNICITY_LABELS = {
        0: "White / Caucasian",
        1: "Black / African",
        2: "Asian",
        3: "South Asian",
        4: "Other"
    }
    
    def __init__(self, checkpoint_path, device="cpu", num_classes=5):
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Initialize multi-task model
        self.model = MultiTaskModel(num_ethnicity_classes=num_classes)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with strict=False to ignore size mismatches in age/gender heads
            # We only care about ethnicity head
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            
            # Check if ethnicity_head loaded correctly
            ethnicity_keys = [k for k in state_dict.keys() if 'ethnicity_head' in k]
            if not ethnicity_keys:
                print("⚠️ Warning: No ethnicity_head keys found in checkpoint")
            else:
                print(f"✓ Ethnicity model loaded from {checkpoint_path}")
                
        except Exception as e:
            print(f"✗ Error loading ethnicity model: {e}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # If already a tensor, ensure correct shape
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)
        
        # Apply transforms
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, image):
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Inference - only use ethnicity head output
        _, _, ethnicity_out = self.model(img_tensor)
        probabilities = torch.softmax(ethnicity_out, dim=1)
        
        # Get prediction
        confidence, predicted_class = torch.max(probabilities, 1)
        class_id = predicted_class.item()
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            self.ETHNICITY_LABELS.get(i, f"Class_{i}"): probabilities[0, i].item()
            for i in range(self.num_classes)
        }
        
        return {
            "ethnicity": self.ETHNICITY_LABELS.get(class_id, "Unknown"),
            "confidence": confidence_score,
            "class_id": class_id,
            "all_probabilities": all_probs
        }
    
    def predict_batch(self, images):
        # Preprocess all images
        img_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.cat(img_tensors, dim=0)
        
        # Inference
        _, _, ethnicity_out = self.model(batch_tensor)
        probabilities = torch.softmax(ethnicity_out, dim=1)
        
        # Get predictions
        confidences, predicted_classes = torch.max(probabilities, 1)
        
        results = []
        for i in range(len(images)):
            class_id = predicted_classes[i].item()
            confidence_score = confidences[i].item()
            
            all_probs = {
                self.ETHNICITY_LABELS.get(j, f"Class_{j}"): probabilities[i, j].item()
                for j in range(self.num_classes)
            }
            
            results.append({
                "ethnicity": self.ETHNICITY_LABELS.get(class_id, "Unknown"),
                "confidence": confidence_score,
                "class_id": class_id,
                "all_probabilities": all_probs
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = EthnicityInference(
        checkpoint_path="checkpoints/ethnicity_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test with a sample image
    from PIL import Image
    
    # Load test image
    test_image = Image.open("test_face.jpg")
    
    # Single prediction
    result = model.predict(test_image)
    print(f"Predicted Ethnicity: {result['ethnicity']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"All probabilities: {result['all_probabilities']}")