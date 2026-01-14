# ∀I-SAGE — Universal / Intelligent Facial Attribute Estimation 🔍🧠


**∀I-SAGE** is a research & demo project that estimates **age**, **gender**, and **ethnicity** from face images in real time. It implements a multi-task deep learning system and provides a live webcam demo for interactive presentations (e.g., conferences).

<img width="856" height="856" alt="VISAGE Logo Concept 3 2" src="https://github.com/user-attachments/assets/af50a2d4-1fcd-4718-a21f-d60cb28081be" />

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

Target latency: **0s per frame**.

---

## Quickstart 🚀

### Requirements
- Python 3.8+
- PyTorch x.x
- OpenCV, torchvision, tqdm, numpy, pandas, etc.
- Streamlit or PyQt5 for demo
- (Optional) NVIDIA GPU + CUDA for training

### Create & activate venv:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Repo layout (concept) 📁
```bash
/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ environment.yaml
├─ src/
│  ├─ model.py
│  ├─ utk_loader.py
│  ├─ training.py
│  └─ infer.py
├─ eval/
│  └─ evaluation.py
├─ demo/
│  └─ video_server.py
└─ checkpoints/
```

## Data & preprocessing 🗂️
-
<img width="1923" height="800" alt="output-onlinepngtools (34)" src="https://github.com/user-attachments/assets/7237f555-5089-4bbf-8fd0-08c9773802eb" />

## Model & training notes 🧩
-

## Evaluation 📊
<img width="1280" height="612" alt="Figure_neu" src="https://github.com/user-attachments/assets/9d1c4991-b2ce-494e-8faa-5399698fb880" />
<img width="2819" height="2374" alt="confusion_matrix_normalized (1)" src="https://github.com/user-attachments/assets/67a79f58-03f4-41ae-bcde-0e1bad59a0e9" />
<img width="936" height="933" alt="Screenshot 2026-01-14 174248" src="https://github.com/user-attachments/assets/2d17d164-45b7-4434-a8cd-b4e5d14f3e2c" />
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/aeae3a1f-174d-4684-97e9-a5467c0ed06f" />


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
Team: Abdelali Oumachi, Aleem Hussain, Ibrahim Jaha

Contact: abdelali.oumachi@study.hs-duesseldorf.de, aleem.hussain@study.hs-duesseldorf.de, ibrahim.jaha@study.hs-duesseldorf.de

# ⚠️ Important: This repository is for research & educational use. Use responsibly and respect privacy & consent.
