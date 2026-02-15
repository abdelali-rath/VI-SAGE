"""Tests for inference API (TorchInference, get_best_inference) and output structure."""

import pytest

try:
    import torch
    from PIL import Image
except Exception:
    pytest.skip("torch/PIL not available", allow_module_level=True)

from src.inference.infer import ETHNICITY_LABELS, TorchInference, get_best_inference
from src.models.models import MultiTaskModel, save_checkpoint


def test_torch_inference_without_checkpoint_returns_valid_structure():
    """TorchInference without checkpoint should still run and return age, gender, ethnicity."""
    inf = TorchInference(checkpoint_path=None, device="cpu", use_fp16=False)
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = inf.predict_from_image(img)
    assert isinstance(result, dict)
    assert "age" in result
    assert "gender" in result
    assert "ethnicity" in result
    assert "label" in result["gender"] and "confidence" in result["gender"]
    assert "label" in result["ethnicity"] and "confidence" in result["ethnicity"]
    assert result["gender"]["label"] in ("male", "female")


def test_get_best_inference_returns_tuple():
    """get_best_inference should return (inference_object, backend_string)."""
    inf, backend = get_best_inference(
        checkpoint_path=None, use_onnx=False, onnx_path=None, use_fp16=False
    )
    assert inf is not None
    assert backend in ("torch-cuda", "torch-cpu")


def test_torch_inference_with_temp_checkpoint(tmp_path):
    """With a real checkpoint file, TorchInference loads it and predicts."""
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5, pretrained=False)
    ckpt = tmp_path / "facesense.pt"
    save_checkpoint(model, str(ckpt))

    inf = TorchInference(checkpoint_path=str(ckpt), device="cpu", use_fp16=False)
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    result = inf.predict_from_image(img)
    assert "age" in result and result["age"] is not None
    assert result["gender"]["label"] in ("male", "female")
    assert result["ethnicity"]["label"] in ETHNICITY_LABELS
