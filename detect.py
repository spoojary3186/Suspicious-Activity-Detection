# detect.py
import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Optional: move sensitive tokens to environment variables (recommended).
# Example: export TELEGRAM_BOT_TOKEN="xxxxx"
BOT_TOKEN = os.getenv("BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"")
CHAT_ID = os.getenv("CHAT_ID = "YOUR_CHAT_ID"")


# TELEGRAM: use telepot (keeps your original choice) but guard initialization
try:
    import telepot
    bot = telepot.Bot(BOT_TOKEN)
except Exception as e:
    bot = None
    print("Telegram setup failed:", e)

# Thread pool for non-blocking telegram sends
THREAD_POOL = ThreadPoolExecutor(max_workers=2)

def _send_telegram_task(full_frame_path, face_path, text):
    """Background task to send telegram attachments and text."""
    if bot is None:
        print("Telegram bot not initialized.")
        return
    try:
        if full_frame_path and Path(full_frame_path).is_file():
            with open(full_frame_path, "rb") as f:
                bot.sendPhoto(CHAT_ID, f)
    except Exception as e:
        print("Telegram full-frame error:", e)
    try:
        if face_path and Path(face_path).is_file():
            with open(face_path, "rb") as f:
                bot.sendPhoto(CHAT_ID, f)
    except Exception as e:
        print("Telegram face error:", e)
    try:
        if text:
            bot.sendMessage(CHAT_ID, text)
    except Exception as e:
        print("Telegram text error:", e)

def send_telegram_async(full_frame_bgr, face_crop_bgr, text):
    """Save temps and dispatch background send to avoid blocking."""
    if bot is None:
        print("Telegram bot not initialized - skipping.")
        return

    import cv2  # import here to avoid top-level dependency issues
    try:
        full_path = "alert_full.jpg"
        cv2.imwrite(full_path, full_frame_bgr)
    except Exception as e:
        print("Failed writing full frame for telegram:", e)
        full_path = None

    face_path = None
    if face_crop_bgr is not None:
        try:
            face_path = "alert_face.jpg"
            cv2.imwrite(face_path, face_crop_bgr)
        except Exception as e:
            print("Failed writing face crop for telegram:", e)
            face_path = None

    # Submit background job
    try:
        THREAD_POOL.submit(_send_telegram_task, full_path, face_path, text)
    except Exception as e:
        print("Failed to submit telegram task:", e)


# =====================================
# Fix Windows incorrect pathlib behavior (keep if needed)
# =====================================
import pathlib
try:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
except Exception:
    # If Not on Windows or fails, ignore
    pass

# Torch imports + SAFE LOADER FIX (best-effort)
try:
    import torch
    import torch.serialization
    try:
        # Attempt to allow custom class loading safely; wrap in try/except
        from models.yolo import Model as YOLOModel  # may fail on some repo layouts
        # If add_safe_globals exists, use it; otherwise ignore
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([YOLOModel])
            print("[INFO] Safe YOLOv5 loader enabled.")
    except Exception:
        pass
except Exception as e:
    print("PyTorch import failed:", e)
    torch = None

import numpy as np
import cv2

# =====================================
# YOLO PATH SETUP
# =====================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# =====================================
# YOLO Dependencies
# =====================================
# Note: these imports depend on repo structure. Keep as-is but guard errors.
try:
    from ultralytics.utils.plotting import Annotator, colors, save_one_box
except Exception:
    # fallback/dummy implementations to avoid crashes if plotting utils missing
    def Annotator(*args, **kwargs):
        class _A:
            def box_label(self, *a, **k): pass
        return _A()
    def colors(*a, **k): return (0, 255, 0)
    def save_one_box(*a, **k): pass

try:
    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (
        LOGGER, Profile, check_file, check_img_size, check_imshow,
        check_requirements, colorstr, cv2 as cv2_util,
        increment_path, non_max_suppression, print_args,
        scale_boxes, strip_optimizer, xyxy2xywh
    )
    from utils.torch_utils import select_device, smart_inference_mode
except Exception as e:
    print("Warning: some YOLO utility imports failed. Make sure repository structure and PYTHONPATH are correct.", e)
    # Provide minimal placeholders to avoid NameError later (they will likely fail at runtime if essential)
    LoadImages = LoadScreenshots = LoadStreams = list
    IMG_FORMATS = []
    VID_FORMATS = []

# =====================================
# Attribute Models (Face/Age/Gender/Mask)
# =====================================
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"

AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

MASK_MODEL = "mask_detector.h5"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_LIST = ['(0-2)', '(4-6)', '(8-14)', '(15-20)', '(21-28)', '(30-40)', '(40-60)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Detection settings
PADDING = 20
FACE_CONF = 0.7
# Use class IDs for weapons (if your dataset maps these IDs to weapon types)
WEAPON_CLASS_IDS = [0, 1, 2, 3, 4]   # example: 0=Grenade,1=knife,2=Pistol,...
WEAPON_CONF = 0.70  # confidence threshold for alerting

# UI/display tuning
DISPLAY_SKIP = 2  # show every (DISPLAY_SKIP+1)th frame; 0 = show every frame

# ===============================
# Load face, age, gender, mask models
# ===============================
def load_attribute_models():
    face = age = gender = mask = None

    try:
        if Path(FACE_MODEL).is_file() and Path(FACE_PROTO).is_file():
            face = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        else:
            print("⚠️ Face detector files missing.")
    except Exception as e:
        print("Face load error:", e)

    try:
        if Path(AGE_MODEL).is_file() and Path(AGE_PROTO).is_file():
            age = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        else:
            print("⚠️ Age model missing.")
    except Exception as e:
        print("Age load error:", e)

    try:
        if Path(GENDER_MODEL).is_file() and Path(GENDER_PROTO).is_file():
            gender = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        else:
            print("⚠️ Gender model missing.")
    except Exception as e:
        print("Gender load error:", e)

    try:
        if Path(MASK_MODEL).is_file():
            from tensorflow.keras.models import load_model
            mask = load_model(MASK_MODEL)
        else:
            print("⚠️ Mask model not found.")
    except Exception as e:
        print("Mask load error:", e)

    return face, age, gender, mask

# =====================================
# Face Detector (OpenCV DNN)
# =====================================
def highlight_face(net, frame, conf=FACE_CONF):
    if net is None:
        return []

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104,117,123], swapRB=True)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0,0,i,2])
        if confidence > conf:
            x1 = int(detections[0,0,i,3] * w)
            y1 = int(detections[0,0,i,4] * h)
            x2 = int(detections[0,0,i,5] * w)
            y2 = int(detections[0,0,i,6] * h)

            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(w-1,x2), min(h-1,y2)

            faces.append((x1, y1, x2, y2))

    return faces

# =====================================
# Age + Gender Prediction
# =====================================
def predict_age_gender(age_net, gender_net, face):
    if age_net is None or gender_net is None:
        return "Unknown", "(Unknown)"

    try:
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[int(np.argmax(gender_preds))]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[int(np.argmax(age_preds))]

        return gender, age

    except Exception:
        return "Unknown", "(Unknown)"

# =====================================
# Mask Prediction
# =====================================
def predict_mask(mask_net, face_roi):
    if mask_net is None:
        return "Unknown"

    try:
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224,224))

        arr = img_to_array(face_resized)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        (mask_prob, no_mask_prob) = mask_net.predict(arr)[0]

        return "NoMask" if no_mask_prob > mask_prob else "Mask"

    except Exception:
        return "Unknown"

# =====================================
# Main YOLO run function
# =====================================
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source=ROOT / "data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.45,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://","rtmp://","http://","https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")

    if is_url and is_file:
        source = check_file(source)

    # Create save folder
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load YOLO
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Load input stream
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Load face/age/gender/mask modules
    faceNet, ageNet, genderNet, maskNet = load_attribute_models()

    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    display_counter = 0  # used for DISPLAY_SKIP

    for path, im, im0s, vid_cap, s in dataset:
        # ===============================
        # Initialize video writer once
        # ===============================
        if 'writer' not in globals():
            writer = None

        if writer is None and vid_cap:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            save_path = str(Path("runs/detect") / "analysis_output.mp4")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            print("[INIT] Writer created →", save_path)

        # Preprocess for YOLO
        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.half() if half else im_tensor.float()
        im_tensor /= 255
        if im_tensor.ndimension() == 3:
            im_tensor = im_tensor[None]

        # YOLO inference
        pred = model(im_tensor)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # What is current frame?
        if webcam:
            p, im0 = path[0], im0s[0].copy()
        else:
            p, im0 = path, im0s.copy()
        p = Path(p)

        # Detect faces
        faces = highlight_face(faceNet, im0)

        # Variables
        detected_weapon = None
        weapon_conf = 0.0
        selected_face_crop = None
        mask_status = "Unknown"
        gender = "Unknown"
        age = "(Unknown)"

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        # ===============================
        # Process YOLO detections
        # ===============================
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    label_name = names[cls] if names and cls < len(names) else str(cls)
                    c = float(conf)

                    # Draw detection
                    label_txt = None if hide_labels else (label_name if hide_conf else f"{label_name} {c:.2f}")
                    annotator.box_label(xyxy, label_txt, color=colors(cls, True))

                    # Check if this is a weapon by class ID
                    if cls in WEAPON_CLASS_IDS and c >= WEAPON_CONF:
                        if c > weapon_conf:
                            detected_weapon = label_name
                            weapon_conf = c

                    # Save crop if requested
                    if save_crop:
                        save_one_box(xyxy, im0, file=save_dir / 'crops' / label_name / f"{p.stem}.jpg", BGR=True)

        # ===============================
        # FACE + AGE + GENDER + MASK (first face only)
        # ===============================
        if len(faces) > 0:
            x1, y1, x2, y2 = faces[0]
            # apply padding + clamp
            x1p = max(0, x1 - PADDING)
            y1p = max(0, y1 - PADDING)
            x2p = min(im0.shape[1]-1, x2 + PADDING)
            y2p = min(im0.shape[0]-1, y2 + PADDING)

            face_crop = im0[y1p:y2p, x1p:x2p].copy()

            if face_crop is not None and face_crop.size != 0:
                gender, age = predict_age_gender(ageNet, genderNet, face_crop)
                mask_status = predict_mask(maskNet, face_crop)

                selected_face_crop = face_crop
                # Re-run mask detection on the EXACT crop that may be sent
                mask_status = predict_mask(maskNet, selected_face_crop)

                # Put text under face box
                label_text = f"{gender}, {age}, {mask_status}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(im0, (x1p, y2p + 3), (x1p + tw + 6, y2p + 20), (0,0,0), cv2.FILLED)
                cv2.putText(im0, label_text, (x1p + 3, y2p + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                box_color = (0,255,0) if mask_status.lower().startswith("mask") else ((0,0,255) if mask_status.lower().startswith("nomask") else (255,255,0))
                cv2.rectangle(im0, (x1p, y1p), (x2p, y2p), box_color, 1)

        # ===============================
        # SEND TELEGRAM ALERT (async)
        # Logic: alert when a weapon (from WEAPON_CLASS_IDS) is detected AND mask_status == "NoMask"
        # (This matches your comment in the original script).
        # ===============================

        
        weapon_flag = detected_weapon is not None
        mask_flag = isinstance(mask_status, str) and mask_status.lower().startswith("mask")
        no_mask_flag = isinstance(mask_status, str) and mask_status.lower().startswith("nomask")

        # --- FINAL ALERT CONDITION ---
        if weapon_flag or mask_flag:
            # BUT DO NOT alert when (no mask AND no weapon)
            if not (no_mask_flag and not weapon_flag):
                message = (
                    "⚠️ Suspicious Activity Detected ⚠️\n"
                    f"Weapon: {detected_weapon}\n"
                    f"Confidence: {weapon_conf:.2f}\n"
                    f"Gender: {gender}\n"
                    f"Age: {age}\n"
                    f"Mask: {mask_status}\n"
                )
                send_telegram_async(im0.copy(), selected_face_crop, message)
        
        # Write processed frame
        if writer is not None:
            writer.write(im0)

        # ===============================
        # Show live feed (throttled)
        # ===============================
        if view_img:
            display_counter += 1
            if DISPLAY_SKIP <= 0 or (display_counter % (DISPLAY_SKIP + 1) == 0):
                cv2.imshow("Live", im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # cleanup
    try:
        THREAD_POOL.shutdown(wait=False)
    except Exception:
        pass


# ===============================
# CLI and main
# ===============================
def parse_opt(source_path):
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default=source_path)
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.45)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    temp_file = Path("temp.txt")
    if not temp_file.exists():
        print("❌ ERROR: temp.txt not found. Create temp.txt and put webcam index (0) or path)")
        sys.exit(1)
    source_path = temp_file.read_text().strip()
    opt = parse_opt(source_path)
    main(opt)
