"""
∀I-SAGE
Streamlit video demo with GPU-accelerated inference.
Run with:
        streamlit run demo/video_server.py
in bash
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.gender_model import GenderInference
from src.age_model import AgeInference
from src.ethnicity_model import EthnicityInference
from torchvision import transforms
import streamlit as st
import cv2
import threading
import time
import torch
import numpy as np
from queue import Queue, Empty
from PIL import Image
import platform

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

# ---------------- GLOBAL QUEUES AND STATE (initialized early) ----------------
gender_queue = Queue(maxsize=16)
gender_results = {}

age_queue = Queue(maxsize=16)
age_results = {}

ethnicity_queue = Queue(maxsize=16)
ethnicity_results = {}

# Temporal smoothing for predictions
prediction_history = {}  # {track_id: {'gender': [], 'age': [], 'ethnicity': []}}
HISTORY_SIZE = 8  # Increased for more stable predictions
MIN_CONFIDENCE_THRESHOLD = 0.4  # Ignore low confidence predictions

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

# ---------------- PREPROCESSING TRANSFORMS ----------------
gender_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

age_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ethnicity_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------- MODEL INITIALIZATION ----------------
try:
    gender_model = GenderInference(
        checkpoint_path="checkpoints/utk_gender_mobilenet.pt",
        device="cpu"
    )
except Exception as e:
    print(f"Warning: Gender model failed to load: {e}")
    gender_model = None

try:
    age_model = AgeInference(
        checkpoint_path="checkpoints/utk_age_mobilenet.pt",
        device="cpu"
    )
except Exception as e:
    print(f"Warning: Age model failed to load: {e}")
    age_model = None

try:
    ethnicity_model = EthnicityInference(
        checkpoint_path="checkpoints/ethnicity_model.pt",
        device="cpu"
    )
except Exception as e:
    print(f"Warning: Ethnicity model failed to load: {e}")
    ethnicity_model = None

# ---------------- WORKER FUNCTIONS ----------------
def smooth_prediction(track_id, pred_type, new_value, new_confidence=None):
    """Smooth predictions over time using history with confidence filtering"""
    if track_id not in prediction_history:
        prediction_history[track_id] = {'gender': [], 'age': [], 'ethnicity': []}
    
    history = prediction_history[track_id][pred_type]
    
    if pred_type == 'age':
        # For age: filter outliers and use weighted average
        history.append(new_value)
        if len(history) > HISTORY_SIZE:
            history.pop(0)
        
        if len(history) < 3:
            return int(new_value)
        
        # Remove outliers (values too far from median)
        ages = np.array(history)
        median_age = np.median(ages)
        std_age = np.std(ages)
        
        # Keep only ages within 1.5 standard deviations
        filtered = ages[np.abs(ages - median_age) <= 1.5 * std_age]
        
        if len(filtered) > 0:
            # Weighted average: newer predictions have more weight
            weights = np.linspace(0.5, 1.0, len(filtered))
            weighted_age = np.average(filtered, weights=weights)
            return int(weighted_age)
        else:
            return int(median_age)
    else:
        # For gender and ethnicity: confidence-weighted voting with threshold
        # Only add predictions with sufficient confidence
        if new_confidence is not None and new_confidence >= MIN_CONFIDENCE_THRESHOLD:
            history.append((new_value, new_confidence))
            if len(history) > HISTORY_SIZE:
                history.pop(0)
        elif new_confidence is None:
            # No confidence provided, assume 1.0
            history.append((new_value, 1.0))
            if len(history) > HISTORY_SIZE:
                history.pop(0)
        
        if len(history) == 0:
            return new_value, new_confidence if new_confidence else 1.0
        
        # Vote by confidence with exponential weighting (newer = more important)
        vote_dict = {}
        for idx, (val, conf) in enumerate(history):
            if val not in vote_dict:
                vote_dict[val] = 0
            # Exponential weight: newer predictions count more
            weight = conf * (1.2 ** idx)
            vote_dict[val] += weight
        
        # Require minimum total confidence before returning result
        total_confidence = sum(vote_dict.values())
        
        best_val = max(vote_dict.items(), key=lambda x: x[1])[0]
        avg_conf = vote_dict[best_val] / len(history)
        
        # Clamp confidence to [0, 1]
        avg_conf = min(1.0, avg_conf)
        
        return best_val, avg_conf

def gender_worker():
    """Process gender predictions asynchronously"""
    while not stop_event.is_set():
        try:
            pil_img, face_id = gender_queue.get(timeout=0.2)
        except Empty:
            continue

        if gender_model is None:
            gender_queue.task_done()
            continue

        try:
            img = gender_transform(pil_img).unsqueeze(0)
            result = gender_model.predict(img)
            
            # Smooth the prediction
            smoothed_label, smoothed_conf = smooth_prediction(
                face_id, 'gender', 
                result["gender"], 
                result["confidence"]
            )
            
            gender_results[face_id] = {
                "gender": smoothed_label,
                "confidence": smoothed_conf
            }
        except Exception as e:
            print(f"Gender worker error: {e}")

        gender_queue.task_done()

def age_worker():
    """Process age predictions asynchronously"""
    while not stop_event.is_set():
        try:
            pil_img, face_id = age_queue.get(timeout=0.2)
        except Empty:
            continue

        if age_model is None:
            age_queue.task_done()
            continue

        try:
            img = age_transform(pil_img).unsqueeze(0)
            result = age_model.predict(img)
            
            # Smooth the age prediction
            smoothed_age = smooth_prediction(face_id, 'age', result["age"])
            
            age_results[face_id] = {"age": smoothed_age}
        except Exception as e:
            print(f"Age worker error: {e}")

        age_queue.task_done()

def ethnicity_worker():
    """Process ethnicity predictions asynchronously"""
    while not stop_event.is_set():
        try:
            pil_img, face_id = ethnicity_queue.get(timeout=0.2)
        except Empty:
            continue

        if ethnicity_model is None:
            ethnicity_queue.task_done()
            continue

        try:
            img = ethnicity_transform(pil_img).unsqueeze(0)
            result = ethnicity_model.predict(img)
            
            # Smooth the prediction
            smoothed_label, smoothed_conf = smooth_prediction(
                face_id, 'ethnicity',
                result["ethnicity"],
                result["confidence"]
            )
            
            ethnicity_results[face_id] = {
                "ethnicity": smoothed_label,
                "confidence": smoothed_conf
            }
        except Exception as e:
            print(f"Ethnicity worker error: {e}")

        ethnicity_queue.task_done()

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
    if platform.system().lower() == "darwin":
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

def load_dnn_detector(proto_path, model_path, try_use_cuda=True):
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError("DNN face model files not found.")

    net = cv2.dnn.readNet(proto_path, model_path)
    system = platform.system().lower()

    if try_use_cuda and has_cv2_cuda() and system != "darwin":
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("DNN: CUDA backend enabled")
            return net
        except Exception:
            pass

    if system == "darwin":
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("DNN: macOS CPU backend enabled")
        except Exception:
            pass
        return net

    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("DNN: CPU fallback backend enabled")
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

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

# Global variables for face tracking with feature matching
tracked_faces = {}  # {track_id: {'box': box, 'last_frame': frame, 'features': embedding, 'history': []}}
next_track_id = 0
current_frame_num = 0
IOU_THRESHOLD = 0.3  # Minimum IoU for same position
MAX_FRAMES_MISSING = 5  # Remove quickly - don't try to re-identify

def assign_track_ids(new_boxes, frame_for_features):
    """Assign stable track IDs using IoU only - simpler and more stable"""
    global next_track_id, tracked_faces, current_frame_num
    
    current_frame_num += 1
    assigned_ids = {}
    used_track_ids = set()
    
    # Try to match new boxes with existing tracked faces using IoU only
    for new_idx, new_box in enumerate(new_boxes):
        best_iou = 0
        best_track_id = None
        
        for track_id, track_data in tracked_faces.items():
            if track_id in used_track_ids:
                continue
            
            old_box = track_data['box']
            iou = calculate_iou(new_box, old_box)
            
            if iou > best_iou and iou >= IOU_THRESHOLD:
                best_iou = iou
                best_track_id = track_id
        
        if best_track_id is not None:
            # Existing face
            assigned_ids[new_idx] = best_track_id
            used_track_ids.add(best_track_id)
            
            tracked_faces[best_track_id] = {
                'box': new_box,
                'last_frame': current_frame_num,
                'history': []
            }
        else:
            # New face - assign new track ID
            assigned_ids[new_idx] = next_track_id
            tracked_faces[next_track_id] = {
                'box': new_box,
                'last_frame': current_frame_num,
                'history': []
            }
            next_track_id += 1
    
    # Remove old tracked faces quickly
    to_remove = []
    for track_id, track_data in tracked_faces.items():
        if current_frame_num - track_data['last_frame'] > MAX_FRAMES_MISSING:
            to_remove.append(track_id)
    
    for track_id in to_remove:
        del tracked_faces[track_id]
        gender_results.pop(track_id, None)
        age_results.pop(track_id, None)
        ethnicity_results.pop(track_id, None)
        prediction_history.pop(track_id, None)
    
    return assigned_ids

# ---------------- THREADS ----------------
def capture_thread_fn(device_index=0):
    global latest_frame
    system = platform.system().lower()

    if system == "windows":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    elif system == "darwin":
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(device_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    
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

        except Exception as e:
            print(f"Detector error: {e}", file=sys.stderr)
            scaled_boxes = []

        with boxes_lock:
            boxes[:] = scaled_boxes
        with labels_lock:
            labels.clear()
            labels.extend([None]*len(scaled_boxes))

        # Assign stable track IDs to detected faces (pass frame for feature extraction)
        track_id_mapping = assign_track_ids(scaled_boxes, f)

        # Enqueue crops for attribute estimation
        for idx, box in enumerate(scaled_boxes):
            track_id = track_id_mapping.get(idx)
            if track_id is None:
                continue
                
            x1,y1,x2,y2,_ = box
            with latest_frame_lock:
                fcopy = None if latest_frame is None else latest_frame.copy()
            if fcopy is None: continue
            
            crop = fcopy[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            
            if run_estimation:
                try:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(crop_rgb)
                    
                    # Submit to all three queues with track_id
                    try:
                        gender_queue.put_nowait((pil.copy(), track_id))
                    except:
                        pass
                    try:
                        age_queue.put_nowait((pil.copy(), track_id))
                    except:
                        pass
                    try:
                        ethnicity_queue.put_nowait((pil.copy(), track_id))
                    except:
                        pass
                except Exception as e:
                    print(f"Crop processing error: {e}")
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
            print(f"Inference worker error: {e}", file=sys.stderr)
            res = None
        with labels_lock:
            if idx < len(labels):
                labels[idx] = res
        inference_queue.task_done()

# ---------------- STARTUP ----------------
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

# Decide parameters based on maximize_fps toggle
if maximize_fps:
    applied_detect_every = max(4, DETECT_EVERY_N_FRAMES * 2)
    applied_detect_width = max(240, int(DETECT_WIDTH * 0.75))
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
    
    # Start capture thread
    t_cap = threading.Thread(target=capture_thread_fn, daemon=True)
    threads.append(t_cap)
    t_cap.start()
    
    # Start detector thread
    if detector_type is not None and detector_obj is not None:
        t_det = threading.Thread(target=detector_thread_fn, args=(detector_type, detector_obj, applied_detect_every, applied_detect_width), daemon=True)
        threads.append(t_det)
        t_det.start()
    
    # Start inference worker if needed
    if applied_run_estimation and inf_runner is not None:
        for i in range(INFERENCE_WORKER_COUNT):
            t_inf = threading.Thread(target=inference_worker_fn, args=(inf_runner,i), daemon=True)
            threads.append(t_inf)
            t_inf.start()
    
    # Start attribute estimation workers
    if applied_run_estimation:
        t_gender = threading.Thread(target=gender_worker, daemon=True)
        threads.append(t_gender)
        t_gender.start()
        
        t_age = threading.Thread(target=age_worker, daemon=True)
        threads.append(t_age)
        t_age.start()
        
        t_ethnicity = threading.Thread(target=ethnicity_worker, daemon=True)
        threads.append(t_ethnicity)
        t_ethnicity.start()

def stop_all():
    stop_event.set()
    time.sleep(0.2)
    
    # Clear all queues
    for q in [inference_queue, gender_queue, age_queue, ethnicity_queue]:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Exception:
                break
    
    # Clear results and history
    gender_results.clear()
    age_results.clear()
    ethnicity_results.clear()
    prediction_history.clear()
    tracked_faces.clear()

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
    
    while not stop_event.is_set():
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.005)
            continue

        annotated = frame.copy()
        with boxes_lock:
            current_boxes = list(boxes)
        with labels_lock:
            current_labels = list(labels)

        # Get track ID mapping for current boxes (pass frame for features)
        track_id_mapping = assign_track_ids(current_boxes, frame) if current_boxes else {}

        if current_boxes:
            for i, b in enumerate(current_boxes):
                try:
                    track_id = track_id_mapping.get(i)
                    if track_id is None:
                        continue
                    
                    x1, y1, x2, y2, score = b
                    
                    # Draw rectangle with consistent color per track_id
                    color_idx = track_id % 6
                    colors = [(0, 255, 0), (255, 0, 255), (255, 255, 0), 
                             (0, 255, 255), (255, 128, 0), (128, 0, 255)]
                    color = colors[color_idx]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    
                    # Prepare text to display
                    text_lines = []
                    y_offset = y1 - 5

                    # Gender
                    if track_id in gender_results:
                        g_label = gender_results[track_id]["gender"]
                        g_conf  = gender_results[track_id]["confidence"]
                        text_lines.append((f"{g_label.capitalize()} ({g_conf:.1%})", (0, 255, 255), y_offset))
                        y_offset -= 22

                    # Age
                    if track_id in age_results:
                        age_value = age_results[track_id]["age"]
                        text_lines.append((f"Age: {int(age_value)}", (0, 200, 255), y_offset))
                        y_offset -= 22

                    # Ethnicity
                    if track_id in ethnicity_results:
                        eth_label = ethnicity_results[track_id]["ethnicity"]
                        eth_conf  = ethnicity_results[track_id]["confidence"]
                        text_lines.append((f"{eth_label} ({eth_conf:.1%})", (255, 150, 0), y_offset))
                    
                    # Adjust if text would go outside image bounds
                    min_y = 15
                    first_text_y = text_lines[0][2] if text_lines else y1
                    if first_text_y < min_y:
                        offset = min_y - first_text_y
                        text_lines = [(text, color, y + offset) for text, color, y in text_lines]
                    
                    # Draw all text lines with background for better readability
                    for text, txt_color, txt_y in reversed(text_lines):
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        # Draw semi-transparent background
                        overlay = annotated.copy()
                        cv2.rectangle(
                            overlay,
                            (x1 - 2, txt_y - text_height - 2),
                            (x1 + text_width + 2, txt_y + baseline + 2),
                            (0, 0, 0),
                            -1
                        )
                        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                        
                        # Draw text
                        cv2.putText(
                            annotated,
                            text,
                            (x1, txt_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            txt_color,
                            2
                        )

                except Exception as e:
                    print(f"Main loop annotation error: {e}")
                    continue
        else:
            cv2.putText(
                annotated,
                "No faces detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        with annotated_lock:
            annotated_frame = annotated

        # Show frame
        video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        # FPS counting
        frame_counter += 1
        now = time.time()
        
        if now - fps_last_time >= STATUS_UPDATE_INTERVAL:
            fps = frame_counter / (now - fps_last_time)
            frame_counter = 0
            fps_last_time = now

            # Build results text
            out_lines = []
            with labels_lock:
                lb = list(labels)
            with boxes_lock:
                bx = list(boxes)
            
            for i in range(len(bx)):
                try:
                    track_id = track_id_mapping.get(i)
                    if track_id is None:
                        continue
                    
                    age = age_results.get(track_id, {}).get("age", "...")
                    gender = gender_results.get(track_id, {}).get("gender", "...")
                    ethnicity = ethnicity_results.get(track_id, {}).get("ethnicity", "...")
                    x1, y1, _, _, _ = bx[i]
                    
                    if age == "..." and gender == "..." and ethnicity == "...":
                        out_lines.append(f"Face @({x1},{y1}) → Analyzing...")
                    else:
                        age_str = str(int(age)) if isinstance(age, (int, float)) else age
                        gender_str = gender.capitalize() if isinstance(gender, str) else gender
                        out_lines.append(f"Face @({x1},{y1}) → {gender_str}, {age_str} yrs, {ethnicity}")
                except Exception as e:
                    print(f"Results display error: {e}")
                    pass

            if out_lines:
                results_box.text("\n".join(out_lines) + f"\n\nFPS: {fps:.1f}")
            else:
                results_box.text(f"FPS: {fps:.1f} | Faces: {len(bx)}")

            status_bar.markdown(f"**Status:** Running | FPS: {fps:.1f} | Faces: {len(bx)} | Detector: {detector_type} | Est: {applied_run_estimation}")

        time.sleep(0.002)

except KeyboardInterrupt:
    stop_all()
    status_bar.markdown("**Status:** Interrupted")