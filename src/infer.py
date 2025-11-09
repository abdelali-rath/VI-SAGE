"""
Inference pipeline with GPU / FP16 / ONNX support for ∀I-SAGE.

Features:
- automatically chooses device = "cuda" if available
- supports optional use_fp16 (autocast) for GPU speedups
- helper to export the PyTorch model to ONNX
- ONNXRuntime-based inference class which will use GPU provider if installed
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Local import (MultiTaskModel + load_checkpoint)
from models import MultiTaskModel, load_checkpoint

# Optional ONNXRuntime
try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except Exception:
    _HAS_ONNXRUNTIME = False

# device detection
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

ETHNICITY_LABELS = ["White", "Black", "Asian", "Indian", "Other"]


class TorchInference:
    """
    Torch-based inference pipeline with optional FP16 (autocast) on GPU.
    """
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None,
                 n_ethnicity: int = 5,
                 use_fp16: bool = False):
        self.device = device or DEFAULT_DEVICE
        self.use_fp16 = use_fp16 and (self.device.startswith("cuda"))
        # instantiate model
        self.model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=n_ethnicity)
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = load_checkpoint(self.model, checkpoint_path, device=self.device)
        else:
            self.model.to(self.device)
            self.model.eval()

    def _preprocess_pil(self, pil: Image.Image) -> torch.Tensor:
        return PREPROCESS(pil).unsqueeze(0).to(self.device)

    def predict_from_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Input: PIL RGB image.
        Returns: dict with bbox-less prediction (age, gender, ethnicity, confidences).
        """
        x = self._preprocess_pil(pil_image)
        self.model.eval()
        with torch.no_grad():
            if self.use_fp16:
                # faster on GPU in many cases
                with torch.cuda.amp.autocast():
                    out = self.model(x)
            else:
                out = self.model(x)

        # parse outputs
        age_val = out["age"].cpu().numpy().item() if out["age"].numel() else None
        g_logits = out["gender_logits"]
        g_probs = torch.softmax(g_logits, dim=1).cpu().numpy()[0]
        gender_idx = int(g_probs.argmax())
        gender_conf = float(g_probs[gender_idx])
        gender_label = "male" if gender_idx == 1 else "female"

        e_logits = out["ethnicity_logits"]
        e_probs = torch.softmax(e_logits, dim=1).cpu().numpy()[0]
        eth_idx = int(e_probs.argmax())
        eth_label = ETHNICITY_LABELS[eth_idx] if eth_idx < len(ETHNICITY_LABELS) else f"cls_{eth_idx}"
        eth_conf = float(e_probs[eth_idx])

        return {
            "age": round(float(age_val), 1) if age_val is not None else None,
            "gender": {"label": gender_label, "confidence": round(gender_conf, 3)},
            "ethnicity": {"label": eth_label, "confidence": round(eth_conf, 3)}
        }


# ---------- ONNX helpers ----------
def export_onnx(checkpoint_path: str,
                onnx_out: str = "model.onnx",
                input_size: Tuple[int, int] = (3, 224, 224),
                opset: int = 13,
                device: Optional[str] = None):
    """
    Export a single forward of the MultiTaskModel (with random heads) to ONNX.
    Loads checkpoint (if present), exports to onnx_out.
    """
    device = device or DEFAULT_DEVICE
    # build model and load weights if given
    model = MultiTaskModel(backbone_name="resnet18", n_ethnicity=5)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_checkpoint(model, checkpoint_path, device="cpu")
    model.eval()
    dummy = torch.randn((1, ) + input_size, dtype=torch.float32)
    # move to cpu for ONNX stable export
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_out,
            opset_version=opset,
            input_names=["input"],
            output_names=["age", "gender_logits", "ethnicity_logits"],
            dynamic_axes={"input": {0: "batch_size"},
                          "age": {0: "batch_size"},
                          "gender_logits": {0: "batch_size"},
                          "ethnicity_logits": {0: "batch_size"}},
            verbose=False
        )
        return onnx_out
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


class ONNXInference:
    """
    Lightweight ONNXRuntime-based inference. If ONNXRuntime with CUDA provider is installed,
    it will use GPU (faster). If not available, falls back to CPU provider.
    """
    def __init__(self, onnx_path: str, provider_preference: Optional[list] = None):
        if not _HAS_ONNXRUNTIME:
            raise RuntimeError("onnxruntime not installed. Install onnxruntime-gpu or onnxruntime.")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{onnx_path} not found")
        self.onnx_path = onnx_path

        # choose execution providers (prefer CUDA if available)
        if provider_preference is None:
            # typical preference: CUDAExecutionProvider, CPUExecutionProvider
            provider_preference = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                provider_preference.append("CUDAExecutionProvider")
            provider_preference.append("CPUExecutionProvider")

        # create session
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=provider_preference)
        except Exception as e:
            # fallback to default providers
            self.sess = ort.InferenceSession(onnx_path)

        # get input name
        self.input_name = self.sess.get_inputs()[0].name

    def predict_from_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        x = PREPROCESS(pil_image).unsqueeze(0).numpy().astype(np.float32)
        # run
        ort_outs = self.sess.run(None, {self.input_name: x})
        # parse outputs (age, gender_logits, ethnicity_logits)
        age = float(ort_outs[0].squeeze().tolist())
        gender_logits = np.array(ort_outs[1])
        gender_probs = softmax_np(gender_logits)[0]
        gender_idx = int(gender_probs.argmax())
        g_conf = float(gender_probs[gender_idx])
        gender_label = "male" if gender_idx == 1 else "female"
        eth_logits = np.array(ort_outs[2])
        eth_probs = softmax_np(eth_logits)[0]
        eth_idx = int(eth_probs.argmax())
        eth_label = ETHNICITY_LABELS[eth_idx] if eth_idx < len(ETHNICITY_LABELS) else f"cls_{eth_idx}"
        e_conf = float(eth_probs[eth_idx])
        return {
            "age": round(age, 1),
            "gender": {"label": gender_label, "confidence": round(g_conf, 3)},
            "ethnicity": {"label": eth_label, "confidence": round(e_conf, 3)}
        }


# small numpy softmax
def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


# Quick convenience factory
def get_best_inference(checkpoint_path: Optional[str] = None,
                       use_onnx: bool = False,
                       onnx_path: Optional[str] = None,
                       use_fp16: bool = True) -> Tuple[Any, str]:
    """
    Returns (inference_object, backend_str). backend_str = 'torch-cuda'|'torch-cpu'|'onnx-cuda'|'onnx-cpu'
    """
    if use_onnx and onnx_path and os.path.exists(onnx_path) and _HAS_ONNXRUNTIME:
        # attempt ONNX GPU if provider present
        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            backend = "onnx-cuda"
        else:
            backend = "onnx-cpu"
        inf = ONNXInference(onnx_path, provider_preference=providers)
        return inf, backend

    # default: torch
    device = DEFAULT_DEVICE
    backend = "torch-cuda" if device.startswith("cuda") else "torch-cpu"
    inf = TorchInference(checkpoint_path=checkpoint_path, device=device, use_fp16=use_fp16)
    return inf, backend
