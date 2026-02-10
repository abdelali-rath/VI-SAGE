# ∀I-SAGE — Universal / Intelligent Facial Attribute Estimation 🔍🧠

**∀I-SAGE** is a research & demo project that estimates **age**, **gender**, and **ethnicity** from face images in real time.  
It implements a multi-task deep learning system and provides a live webcam demo for interactive presentations (e.g. conferences).

<img width="856" height="856" alt="VISAGE Logo Concept 3 2" src="https://github.com/user-attachments/assets/af50a2d4-1fcd-4718-a21f-d60cb28081be" />

---

## Features ✨
- **Realtime demo**: OpenCV + Streamlit webcam app with overlaid predictions.
- **Multiple attributes**: age (regression + binned ranges), gender (binary), ethnicity (5-way classification).
- **Task‑specific training scripts**: separate training for age, gender and ethnicity on UTKFace.
- **GPU‑accelerated inference**: optional CUDA / TorchScript / ONNX where available.
- **Config‑driven setup**: centralized `config/default.yaml` for paths, checkpoints, and demo settings.
- **Ethics & privacy focus**: visible disclaimers and guidance for responsible use.

---

## Demo (what to expect) 🎛️
- Live webcam feed with face bounding boxes.
- For each detected face:
  - Estimated age (years) and age range (e.g. “Teen”, “Erwachsener”).
  - Predicted gender with confidence.
  - Predicted ethnicity class with confidence.
- Always‑visible ethics / disclaimer box inside the UI.

Target latency: **interactive realtime** on a modern laptop (exact FPS depends on hardware and resolution).

---

## Quickstart 🚀

### Requirements
- Python **3.8+**
- PyTorch, torchvision
- OpenCV (`opencv-python`)
- Streamlit
- `tqdm`, `numpy`, `pandas`, `Pillow`, etc.  
  → All listed in `requirements.txt`.

Optional:
- NVIDIA GPU + CUDA for faster training / inference.

### Create & activate a virtualenv

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the live Streamlit demo 🎥

From the project root:

```bash
streamlit run app/video_server.py
```

Notes:
- Allow the browser / OS to access your webcam.
- On macOS you may need to grant camera permissions to your terminal / Python.
- By default models are loaded from `checkpoints/` (see “Checkpoints & configuration” below).

---

## Training scripts 🏋️‍♀️

All training scripts expect the **UTKFace** dataset in `data/UTKFace`.  
Filenames should follow the pattern `age_gender_race_*.jpg`, e.g. `25_0_2_201612312359.jpg`.

### Train age regression

Script: `src/training/train_age/age_prediction.py`

```bash
python src/training/train_age/age_prediction.py
```

Saves weights to `checkpoints/utk_age_model.pt`.

### Train gender classification

Script: `src/training/train_gender/train_gender_utk.py`

```bash
python src/training/train_gender/train_gender_utk.py
```

Saves weights to `checkpoints/utk_gender_model.pt`.

### Train ethnicity classification

Script: `src/training/train_ethnicity/train_ethnicity.py`

```bash
python -m src.training.train_ethnicity.train_ethnicity
```

Uses a shared `MultiTaskModel` backbone and trains the ethnicity head, saving
to `checkpoints/utk_ethnicity_model.pt`.

---

## Checkpoints & configuration ⚙️

- **Model checkpoints** are expected in `checkpoints/`:
  - `utk_age_model.pt`
  - `utk_gender_model.pt`
  - `utk_ethnicity_model.pt`
  - Optional TorchScript / ONNX variants used by the fast demo.
- **Global configuration** lives in `config/default.yaml` and includes:
  - project metadata,
  - dataset and checkpoint paths,
  - training hyperparameters for age/gender/ethnicity,
  - demo options (camera resolution, detection thresholds, smoothing, tracking),
  - debug flags.

You can adjust resolutions, detection confidence, or paths in the YAML file instead
of editing Python constants in `app/video_server.py` and the training scripts.

---

## Repository layout 📁

```bash
/
├─ app/
│  └─ video_server.py          # Streamlit webcam demo
├─ config/
│  └─ default.yaml             # Central configuration
├─ src/
│  ├─ models/                  # Age / gender / ethnicity models and multi‑task backbone
│  ├─ inference/
│  │  └─ infer.py              # Helper to pick best inference backend
│  └─ training/
│     ├─ train_age/
│     │  └─ age_prediction.py
│     ├─ train_gender/
│     │  ├─ train_gender_utk.py
│     │  └─ utk_loader.py
│     └─ train_ethnicity/
│        ├─ train_ethnicity.py
│        └─ utk_loader.py
├─ tests/
│  └─ check_load_models.py     # Simple sanity check for model loading
├─ checkpoints/                # (created at runtime) trained weights
├─ data/
│  └─ UTKFace/                 # expected location of UTKFace images
├─ README.md
└─ requirements.txt
```

---

## Ethics & limitations ⚠️🚫

Predicting sensitive attributes (especially **ethnicity**) is controversial and can be harmful if misused.
This project is strictly for **research and educational** purposes.

We adopt the following safeguards:
- **Consent & transparency**: only run the demo with informed consent of participants.
- **No decision making**: never use predictions for decisions about individuals.
- **Bias awareness**: models trained on UTKFace inherit dataset biases and may perform worse on
  under‑represented groups.
- **Privacy**: do not store or publish identifiable face images or recordings without explicit consent.
- **Visible disclaimers**: the demo clearly labels results as noisy estimates with uncertainty.

Any publication or report using this repository **must** include a detailed ethics & impact discussion.

---

## Team & contact 📬

- **Project**: ∀I-SAGE  
- **Team**: Abdelali Oumachi, Aleem Hussain, Ibrahim Jaha  
- **Contact**:  
  - abdelali.oumachi@study.hs-duesseldorf.de  
  - aleem.hussain@study.hs-duesseldorf.de  
  - ibrahim.jaha@study.hs-duesseldorf.de

---

## Legal notice

This repository is provided **for research & educational use only**.  
Use it responsibly, respect privacy and consent, and comply with all applicable laws and regulations.
