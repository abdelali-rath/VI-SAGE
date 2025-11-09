"""
Lightweight multi-task model (age, gender, ethnicity).
This file provides:
- build_base_backbone(...) -> (feature_extractor, feature_dim)
- MultiTaskModel(backbone_name, n_ethnicity_classes)
- checkpoints
"""


from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


def build_base_backbone(name: str = "resnet18", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Return (feature_extractor_module, feature_dim).
    feature_extractor expects input [B,3,H,W] and returns [B,feature_dim].
    """
    if name != "resnet18":
        raise ValueError("Only 'resnet18' supported in this minimal implementation.")

    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        backbone = models.resnet18(pretrained=pretrained)

    # strip final fc, keep feature extractor returning (B, C, 1, 1)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    feature_dim = backbone.fc.in_features
    return feature_extractor, feature_dim


class MultiTaskModel(nn.Module):
    """
    Shared backbone + three heads:
      - age_head: regression (single value)
      - gender_head: binary logits
      - ethnicity_head: multi-class logits
    """

    def __init__(self, backbone_name: str = "resnet18", n_ethnicity: int = 5, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_base_backbone(backbone_name, pretrained=pretrained)
        self.feat_dim = feat_dim

        # heads
        self.age_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # regression (years)
        )

        self.gender_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # logits for 2 classes
        )

        self.ethnicity_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_ethnicity)  # logits for ethnicity classes
        )

    def forward(self, x):
        """
        x: [B,3,H,W]
        returns dict with raw outputs (no activation)
        """
        feats = self.backbone(x)  # [B, C, 1, 1]
        age = self.age_head(feats)            # [B,1]
        gender_logits = self.gender_head(feats)  # [B,2]
        eth_logits = self.ethnicity_head(feats)  # [B,n_eth]

        return {
            "age": age.squeeze(1),          # [B]
            "gender_logits": gender_logits, # [B,2]
            "ethnicity_logits": eth_logits  # [B,n_eth]
        }


# Simple checkpoint helpers
def save_checkpoint(model: nn.Module, path: str):
    torch.save({"model_state": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()
    return model
