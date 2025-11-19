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
from queue import Queue, Empty
from PIL import Image

# inference
from src.infer import get_best_inference

# ---------------- CONFIG (defaults) ----------------
CAP_WIDTH = 1280
CAP_HEIGHT = 720

DETECT_WIDTH = 320              # default detection resolution (smaller = faster)
DETECT_CONF = 0.5
DETECT_EVERY_N_FRAMES = 3       # default detection frequency

INFERENCE_WORKER_COUNT = 1
CHECKPOINT_PATH = "checkpoints/facesense_debug.pt"
USE_ONNX = False
ONNX_PATH = "checkpoints/facesense.onnx"
USE_FP16 = True

DNN_PROTO = "models/dnn/deploy.prototxt"
DNN_MODEL = "models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel"

STATUS_UPDATE_INTERVAL = 0.8    # seconds - UI update cadence for FPS/results

# ---------------- SHARED STATE ----------------
latest_frame = None
latest_frame_lock = threading.Lock()

annotated_frame = None
annotated_lock = threading.Lock()

boxes = []
boxes_lock = threading.Lock()

labels = []
labels_lock = threading.Lock()

inference_queue = Queue(maxsize=16)
stop_event = threading.Event()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="∀I-SAGE — Fast Live Demo", layout="wide")
st.title("∀I-SAGE — Fast Live Webcam Demo (720p)")

col_video, col_right = st.columns([3,1])
video_placeholder = col_video.empty()

col_right.subheader("Controls")
start_button = col_right.button("▶ Start")
stop_button  = col_right.button("⏹ Stop")

# toggles
run_estimation = col_right.checkbox("Enable attribute estimation (age/gender/ethnicity)", value=False)
maximize_fps = col_right.checkbox("Maximize FPS (aggressive)", value=False)

col_right.markdown("### Results")
results_box = col_right.empty()
col_right.markdown("### Notes")
col_right.markdown(
    "- Resolution: 1280×720\n"
    "- Detector: OpenCV DNN res10_300x300_ssd (fast) with Haar fallback.\n"
    "- Inference: asynchronous on GPU if enabled."
)

status_bar = st.empty()

# ---------------- HELPERS ----------------
def has_cv2_cuda():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def load_dnn_detector(proto_path, model_path, try_use_cuda=True):
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError("DNN face model files not found.")
    net = cv2.dnn.readNet(proto_path, model_path)
    if try_use_cuda and has_cv2_cuda():
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
    return net

def detect_dnn(net, frame_bgr, conf_threshold=0.5, in_size=300):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (in_size, in_size),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()
    res = []
    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        if score < conf_threshold:
            continue
        x1 = int(out[0, 0, i, 3] * w)
        y1 = int(out[0, 0, i, 4] * h)
        x2 = int(out[0, 0, i, 5] * w)
        y2 = int(out[0, 0, i, 6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        res.append((x1,y1,x2,y2,score))
    return res

def detect_haar(cascade, frame_bgr, scale_factor=1.1, min_neighbors=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(60,60))
    res = []
    for (x,y,w,h) in faces:
        x1, y1, x2, y2 = x, y, x+w, y+h
        res.append((x1,y1,x2,y2, None))
    return res

# ---------------- THREADS ----------------
def capture_thread_fn(device_index=0):
    global latest_frame
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    # reduce internal buffer if supported
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.002)
            continue
        frame = cv2.flip(frame, 1)
        with latest_frame_lock:
            latest_frame = frame
        # very small sleep to keep CPU friendly
        time.sleep(0.002)
    cap.release()

def detector_thread_fn(detector_type, detector_obj, detect_every, detect_width):
    detect_counter = 0
    while not stop_event.is_set():
        with latest_frame_lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            time.sleep(0.002)
            continue

        detect_counter += 1
        if (detect_counter % detect_every) != 0:
            time.sleep(0.002)
            continue

        scaled_boxes = []
        try:
            if detector_type == "dnn":
                h, w = f.shape[:2]
                scale = detect_width / float(w) if w > detect_width else 1.0
                small = cv2.resize(f, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else f
                boxes_small = detect_dnn(detector_obj, small, conf_threshold=DETECT_CONF, in_size=300)
                for (x1,y1,x2,y2,score) in boxes_small:
                    if scale != 1.0:
                        x1 = int(x1 / scale); x2 = int(x2 / scale)
                        y1 = int(y1 / scale); y2 = int(y2 / scale)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(f.shape[1]-1, x2), min(f.shape[0]-1, y2)
                    if x2 <= x1 or y2 <= y1: continue
                    scaled_boxes.append((x1,y1,x2,y2,score))

            elif detector_type == "haar":
                boxes_h = detect_haar(detector_obj, f, scale_factor=1.1, min_neighbors=5)
                for (x1,y1,x2,y2,_) in boxes_h:
                    scaled_boxes.append((x1,y1,x2,y2, None))

            elif detector_type == "mtcnn":
                try:
                    dets, _ = detector_obj.detect(f)
                    if dets is not None:
                        for d in dets:
                            x1,y1,x2,y2 = [int(v) for v in d]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(f.shape[1]-1, x2), min(f.shape[0]-1, y2)
                            if x2 <= x1 or y2 <= y1: continue
                            scaled_boxes.append((x1,y1,x2,y2, None))
                except Exception:
                    pass

        except Exception as e:
            print("Detector error:", e, file=sys.stderr)
            scaled_boxes = []

        with boxes_lock:
            boxes[:] = scaled_boxes
        with labels_lock:
            labels.clear(); labels.extend([None]*len(scaled_boxes))

        # enqueue crops if estimation enabled
        for idx, box in enumerate(scaled_boxes):
            x1,y1,x2,y2,_ = box
            with latest_frame_lock:
                fcopy = None if latest_frame is None else latest_frame.copy()
            if fcopy is None: continue
            crop = fcopy[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            # only enqueue if estimation enabled
            if run_estimation:
                try:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(crop_rgb)
                    try:
                        inference_queue.put_nowait((pil, idx))
                    except Exception:
                        pass
                except Exception:
                    continue
        time.sleep(0.001)

def inference_worker_fn(inf_runner, worker_id=0):
    while not stop_event.is_set():
        try:
            pil, idx = inference_queue.get(timeout=0.2)
        except Empty:
            continue
        try:
            res = inf_runner.predict_from_image(pil)
        except Exception as e:
            print("Inference worker error:", e, file=sys.stderr)
            res = None
        with labels_lock:
            if idx < len(labels):
                labels[idx] = res
        inference_queue.task_done()

# ---------------- STARTUP (choose detector, optionally inference) ----------------
detector_type = None
detector_obj = None
det_msg = ""
inf_msg = ""

# Prefer DNN, fallback to Haar
try:
    detector_obj = load_dnn_detector(DNN_PROTO, DNN_MODEL, try_use_cuda=True)
    detector_type = "dnn"
    det_msg = "DNN loaded"
except Exception as e:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            detector_obj = cv2.CascadeClassifier(cascade_path)
            detector_type = "haar"
            det_msg = "Haar cascade loaded (fallback)"
        else:
            detector_type = None
            det_msg = f"No detector available: {e}"
    except Exception as ex:
        detector_type = None
        det_msg = f"No detector available: {ex}"

# Decide parameters based on maximize_fps toggle (read before starting threads)
if maximize_fps:
    applied_detect_every = max(4, DETECT_EVERY_N_FRAMES * 2)
    applied_detect_width = max(240, int(DETECT_WIDTH * 0.75))
    # force disable estimation in maximize mode
    if run_estimation:
        st.warning("Maximize FPS enabled: attribute estimation will be disabled for max speed.")
    applied_run_estimation = False
else:
    applied_detect_every = DETECT_EVERY_N_FRAMES
    applied_detect_width = DETECT_WIDTH
    applied_run_estimation = run_estimation

# Prepare inference runner if estimation enabled
inf_runner = None
if applied_run_estimation:
    try:
        inf_runner, backend_str = get_best_inference(checkpoint_path=CHECKPOINT_PATH,
                                                     use_onnx=USE_ONNX, onnx_path=ONNX_PATH,
                                                     use_fp16=USE_FP16)
        inf_msg = f"Inference backend: {backend_str}"
    except Exception as e:
        inf_runner = None
        inf_msg = f"Inference init failed: {e}"
else:
    inf_msg = "Estimation disabled"

status_bar.markdown(f"**Status:** {det_msg} | {inf_msg} | Detector: {detector_type}")

threads = []

def start_all():
    stop_event.clear()
    t_cap = threading.Thread(target=capture_thread_fn, daemon=True); threads.append(t_cap); t_cap.start()
    if detector_type is not None and detector_obj is not None:
        t_det = threading.Thread(target=detector_thread_fn, args=(detector_type, detector_obj, applied_detect_every, applied_detect_width), daemon=True); threads.append(t_det); t_det.start()
    if applied_run_estimation and inf_runner is not None:
        for i in range(INFERENCE_WORKER_COUNT):
            t_inf = threading.Thread(target=inference_worker_fn, args=(inf_runner,i), daemon=True); threads.append(t_inf); t_inf.start()

def stop_all():
    stop_event.set()
    time.sleep(0.2)
    while not inference_queue.empty():
        try:
            inference_queue.get_nowait(); inference_queue.task_done()
        except Exception:
            break

if start_button:
    start_all()
    status_bar.markdown("**Status:** Running (capture+detector started)")

if stop_button:
    stop_all()
    status_bar.markdown("**Status:** Stopped by user")
    with annotated_lock:
        if annotated_frame is not None:
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

# ---------------- MAIN RENDER LOOP ----------------
try:
    frame_counter = 0
    fps_last_time = time.time()
    last_status_update = 0.0
    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.005); continue

        annotated = frame.copy()
        with boxes_lock:
            current_boxes = list(boxes)
        with labels_lock:
            current_labels = list(labels)

        if current_boxes:
            for i, b in enumerate(current_boxes):
                try:
                    x1,y1,x2,y2,score = b
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)
                    if applied_run_estimation and i < len(current_labels) and current_labels[i] is not None:
                        r = current_labels[i]
                        age = r.get("age", "?")
                        gender = r.get("gender", {}).get("label") if isinstance(r.get("gender"), dict) else r.get("gender")
                        eth = r.get("ethnicity", {}).get("label") if isinstance(r.get("ethnicity"), dict) else r.get("ethnicity")
                        txt = f"{age}, {gender}, {eth}"
                        cv2.putText(annotated, txt, (x1, max(y1-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                except Exception:
                    continue
        else:
            cv2.putText(annotated, "No faces detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        with annotated_lock:
            annotated_frame = annotated

        # show frame
        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        # FPS counting (count frames every loop)
        frame_counter += 1
        now = time.time()
        # update UI at STATUS_UPDATE_INTERVAL
        if now - fps_last_time >= STATUS_UPDATE_INTERVAL:
            fps = frame_counter / (now - fps_last_time)
            frame_counter = 0
            fps_last_time = now

            # build results text
            out_lines = []
            with labels_lock:
                lb = list(labels)
            with boxes_lock:
                bx = list(boxes)
            for i, lab in enumerate(lb):
                if lab is None:
                    out_lines.append(f"Face {i}: detecting..." if applied_run_estimation else f"Face {i}: no attributes")
                    continue
                try:
                    age = lab.get("age", "?")
                    gender = lab.get("gender", {}).get("label") if isinstance(lab.get("gender"), dict) else lab.get("gender")
                    eth = lab.get("ethnicity", {}).get("label") if isinstance(lab.get("ethnicity"), dict) else lab.get("ethnicity")
                    x1,y1,x2,y2,_ = bx[i]
                    out_lines.append(f"Face {i} @({x1},{y1}) → Age: {age}, Gender: {gender}, Eth: {eth}")
                except Exception:
                    continue

            if out_lines:
                results_box.text("\n".join(out_lines) + f"\n\nFPS: {fps:.1f} | Faces: {len(bx)}")
            else:
                results_box.text(f"FPS: {fps:.1f} | Faces: {len(bx)}")

            status_bar.markdown(f"**Status:** Running | FPS: {fps:.1f} | Faces: {len(bx)} | Detector: {detector_type} | Est: {applied_run_estimation}")

        # tiny yield
        time.sleep(0.002)

except KeyboardInterrupt:
    stop_all()
    status_bar.markdown("**Status:** Interrupted")