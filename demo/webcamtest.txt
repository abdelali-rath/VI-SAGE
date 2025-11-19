"""
∀I-SAGE
Streamlit video demo with GPU-accelerated inference.
Run with:
        streamlit run demo/video_server.py
in bash
"""


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import threading
import time
import torch
import numpy as np
from queue import Queue
import io
from facenet_pytorch import MTCNN
# Import inference system (GPU/ONNX compatible)
from src.infer import get_best_inference
from src.models import MultiTaskModel



# Streamlit UI setup
st.set_page_config(page_title="∀I-SAGE Live Demo", layout="wide")

# Title at top
st.title("∀I-SAGE – Live Demo")

# Main layout: video + right panel
col_video, col_right = st.columns([3,1])
frame_placeholder = col_video.empty()

# Right panel: buttons, info, ethics
start = col_right.button("▶ Start Demo")
stop_btn = col_right.button("⏹ Stop Demo")

col_right.subheader("Demo Info")
results_placeholder = col_right.empty()

col_right.markdown("### Ethics Note")
col_right.markdown(
    "This demo is for research purposes. Age, gender, and ethnicity estimations "
    "are approximate and should not be used for real-world decisions."
)

# Bottom status bar
status_placeholder = st.empty()
status_placeholder.markdown("<p style='text-align:center'>Status: Idle</p>", unsafe_allow_html=True)


# Globals
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
stop_stream = False


# Webcam + inference loop
if start:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    status_placeholder.markdown("<p style='text-align:center'>Status: Camera & Inference Running</p>", unsafe_allow_html=True)

    while cap.isOpened() and not stop_stream:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        annotated = frame.copy()
        boxes, _ = mtcnn.detect(frame)
        face_infos = []
        num_faces = 0

        if boxes is not None:
            num_faces = len(boxes)
            for box in boxes:
                try:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Draw bounding box only
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    face_crop = frame[y1:y2, x1:x2].copy()
                    if face_crop.size == 0:
                        continue
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    results = get_best_inference(face_crop_rgb, device=device)
                    face_infos.append(
                        f"Face ({x1},{y1}) → Age: {results.get('age','?')}, "
                        f"Gender: {results.get('gender','?')}, "
                        f"Ethnicity: {results.get('ethnicity','?')}"
                    )
                except Exception:
                    continue

        # Compute FPS
        curr_time = time.time()
        fps = 1 / max((curr_time - prev_time), 1e-6)
        prev_time = curr_time
        fps_text = f"FPS: {fps:.1f} | Faces detected: {num_faces}"

        # Update UI
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        results_placeholder.text("\n".join(face_infos) + "\n\n" + fps_text)
        status_placeholder.markdown(f"<p style='text-align:center'>Status: Running | {fps:.1f} FPS | {num_faces} face(s)</p>", unsafe_allow_html=True)

        # Stop button check
        if stop_btn:
            stop_stream = True
            cap.release()
            cv2.destroyAllWindows()
            status_placeholder.markdown("<p style='text-align:center'>Status: Demo Stopped</p>", unsafe_allow_html=True)
            st.warning("Demo stopped.")
            break

        time.sleep(0.03)  # ~30 FPS

# Idle status
if not start:
    status_placeholder.markdown("<p style='text-align:center'>Status: Idle</p>", unsafe_allow_html=True)