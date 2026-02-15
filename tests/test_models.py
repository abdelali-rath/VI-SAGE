"""Tests for model architectures and checkpoint helpers."""

import pytest

try:
    import torch
except Exception:
    pytest.skip("torch not available (e.g. broken numpy)", allow_module_level=True)

from src.models.age_model import AgeModel
from src.models.gender_model import GenderNet
from src.models.models import (
    MultiTaskModel,
    build_base_backbone,
    load_checkpoint,
    save_checkpoint,
)


def test_age_model_forward_shape(sample_image_tensor):
    model = AgeModel(pretrained=False)
    model.eval()
    out = model(sample_image_tensor)
    assert out.shape == (sample_image_tensor.size(0),)


def test_gender_net_forward_shape(sample_image_tensor):
    model = GenderNet()
    model.eval()
    out = model(sample_image_tensor)
    assert out.shape == (sample_image_tensor.size(0), 2)


def test_multi_task_model_forward_shape(sample_image_tensor):
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5, pretrained=False)
    model.eval()
    out = model(sample_image_tensor)
    assert "age" in out and "gender_logits" in out and "ethnicity_logits" in out
    assert out["age"].shape == (sample_image_tensor.size(0),)
    assert out["gender_logits"].shape == (sample_image_tensor.size(0), 2)
    assert out["ethnicity_logits"].shape == (sample_image_tensor.size(0), 5)


def test_build_base_backbone():
    backbone, feat_dim = build_base_backbone("resnet18", pretrained=False)
    assert feat_dim == 512
    x = torch.rand(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, 512, 1, 1)


def test_build_base_backbone_unsupported_raises():
    with pytest.raises(ValueError, match="Only 'resnet18'"):
        build_base_backbone("resnet50", pretrained=False)


def test_save_and_load_checkpoint(temp_checkpoint_dir):
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5, pretrained=False)
    path = temp_checkpoint_dir / "test.pt"
    save_checkpoint(model, str(path))
    assert path.exists()

    model2 = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5, pretrained=False)
    load_checkpoint(model2, str(path), device="cpu")
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
