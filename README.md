# ∀I-SAGE — Universal / Intelligent Facial Attribute Estimation 🔍🧠


**∀I-SAGE** is a research & demo project that estimates **age**, **gender**, and **ethnicity** from face images in real time. It implements a multi-task deep learning system and provides a live webcam demo for interactive presentations (e.g., conferences).

<img width="1000" height="1000" alt="VISAGE Logo Concept 3" src="https://github.com/user-attachments/assets/2d0ecbd7-16dc-4db0-abca-355e720a79da" />

---

## Features ✨
- Multi-task model: **age** (regression / ordinal), **gender** (binary), **ethnicity** (multi-class)  
- Transfer learning on modern CNN backbones
- Fast realtime demo (OpenCV + Streamlit / PyQt + TorchScript) with overlayed predictions
- Built-in ethics & privacy guidance

---

## Demo (what to expect) 🎛️
- Webcam feed + face bounding box  
- Predicted age (years)
- Predicted gender
- Predicted ethnicity class
- Small ethics/disclaimer box

Target latency: **x s per frame**.

---

## Quickstart 🚀

### Requirements
- Python 3.8+
- PyTorch x.x
- OpenCV, torchvision, tqdm, numpy, pandas, etc.
- Streamlit or PyQt5 for demo
- (Optional) NVIDIA GPU + CUDA for training

Create & activate venv:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Repo layout (concept) 📁
```bash
/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ resnet_baseline.yaml
├─ data/
│  └─ DATA_CARD.md
├─ src/
│  ├─ models/
│  ├─ datasets/
│  ├─ training/
│  └─ utils/
├─ train/
│  └─ train.py
├─ eval/
│  └─ evaluate.py
├─ demo/
│  ├─ infer_webcam.py
│  └─ streamlit_demo.py
├─ checkpoints/
└─ notebooks/
```

## Data & preprocessing 🗂️
-

## Model & training notes 🧩
-

## Evaluation 📊
-

## Ethics & limitations ⚠️🚫

Predicting sensitive attributes (e.g., ethnicity) is controversial and can be harmful if misused. We adopt strict safeguards:
1. Use publicly licensed datasets and include a detailed data card.
2. Show confidence & short disclaimers; avoid deterministic language like “you are X”.
3. Do not release raw face images or identifiable data.
4. Use the demo only for research/education; attendees can opt out.
5. Include a visible ethics note on all demo screens.
The final report must include a thorough ethics & impact discussion.

## Team & contact 📬

Project: ∀I-SAGE
Team: Abdelali Oumachi, Aleem Hussein, Ibrahim Jaha
Contact: abdelali.oumachi@study.hs-duesseldorf.de, aleem.hussein@study.hs-duesseldorf.de, ibrahim.jaha@study.hs-duesseldorf.de

# ⚠️ Important: This repository is for research & educational use. Use responsibly and respect privacy & consent.
