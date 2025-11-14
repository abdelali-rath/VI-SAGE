"""
∀I-SAGE
Streamlit video demo with GPU-accelerated inference.
Run with:
        streamlit run demo/video_server.py
in bash
"""

import sys, os
import time
import cv2
import streamlit as st
import torch

from ultralytics import YOLO
yolo_face = YOLO("yolov8n.pt")  # small and fast face-capable model

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.infer import get_best_inference
from src.models import MultiTaskModel


# Streamlit UI setup
st.set_page_config(page_title="∀I-SAGE Live Demo", layout="wide")

# Title at top
st.title("∀I-SAGE – Live Demo")

if "running" not in st.session_state:
    st.session_state.running = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

col_video, col_right = st.columns([3, 1])
frame_placeholder = col_video.empty()

# Right panel: buttons, info, ethics
start_btn = col_right.button("▶ Start Demo")
stop_btn = col_right.button("⏹ Stop Demo")

col_right.subheader("Demo Info")
results_placeholder = col_right.empty()

col_right.markdown("### Ethics Note")
col_right.markdown(
    "This demo is for research purposes. Age, gender, and ethnicity estimations "
    "are approximate and should not be used for real-world decisions."
)

status_placeholder = st.empty()
status_placeholder.markdown("<p style='text-align:center'>Status: Idle</p>", unsafe_allow_html=True)


# Start / Stop logic
if start_btn:
    st.session_state.running = True
    status_placeholder.markdown(
        "<p style='text-align:center'>Status: Camera & Inference Running</p>",
        unsafe_allow_html=True
    )

if stop_btn:
    st.session_state.running = False
    status_placeholder.markdown(
        "<p style='text-align:center'>Status: Demo Stopped</p>",
        unsafe_allow_html=True
    )


# Main loop: camera + YOLO face detection + inference
if st.session_state.running:

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while st.session_state.running:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        annotated = frame.copy()
        h, w = frame.shape[:2]
        face_infos = []
        num_faces = 0

        # Run YOLO face detection on the frame
        results = yolo_face(frame, verbose=False)

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            num_faces = len(boxes)

            for (x1, y1, x2, y2) in boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # YOLO face boxes are often too wide; we tighten horizontally
                bw = x2 - x1
                bh = y2 - y1

                target_ratio = 0.75   # width ≈ 75% of height for a realistic face crop
                target_width = int(bh * target_ratio)

                extra = bw - target_width

                if extra > 0:
                    x1 += extra // 2
                    x2 -= extra // 2

                # Make sure the box stays inside the image
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Draw final bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop face region for attribute inference
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue

                try:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                except:
                    continue

                # Run age/gender/ethnicity model
                try:
                    pred = get_best_inference(crop_rgb, device=device)
                    face_infos.append(
                        f"Face ({x1},{y1}) → Age: {pred.get('age','?')}, "
                        f"Gender: {pred.get('gender','?')}, "
                        f"Ethnicity: {pred.get('ethnicity','?')}"
                    )
                except:
                    face_infos.append("Prediction error")
                    continue

        # FPS calculation
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # Update the video feed
        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        # Display predictions
        results_placeholder.text(
            "\n".join(face_infos) +
            f"\n\nFPS: {fps:.1f} | Faces detected: {num_faces}"
        )

        # Update status bar
        status_placeholder.markdown(
            f"<p style='text-align:center'>Status: Running | {fps:.1f} FPS | {num_faces} face(s)</p>",
            unsafe_allow_html=True
        )

        time.sleep(0.03)  # ~30 FPS

    cap.release()
