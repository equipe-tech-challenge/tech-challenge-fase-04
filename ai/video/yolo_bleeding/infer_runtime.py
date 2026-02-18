import os
from pathlib import Path
from typing import Dict, Any
from collections import deque

import cv2
from ultralytics import YOLO


CLASS_NAMES = [
    "Baby",
    "Blood",
    "Gauze",
    "Hand",
    "Person",
    "RedGauze",
    "StainedGauze",
    "StainedHand"
]


def load_model() -> YOLO:
    model_path = Path("ai/video/yolo_bleeding/model.pt")
    if not model_path.exists():
        return None
    return YOLO(str(model_path))

def compute_total_bbox_area(boxes_xyxy) -> float:
    total = 0.0
    for (x1, y1, x2, y2) in boxes_xyxy:
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        total += w * h
    return total


def classify_blood_level(blood_ratio: float) -> str:
    """
      < 1%   -> small
      < 5%   -> medium
      >= 5%  -> large
    """
    if blood_ratio < 0.01:
        return "blood_small"
    elif blood_ratio < 0.05:
        return "blood_medium"
    else:
        return "blood_large"


def detect_pooling(history: deque) -> bool:
    """
    pooling = crescimento consistente do blood_ratio.
    """
    if len(history) < 6:
        return False

    first = history[0]
    last = history[-1]

    if first <= 0.0001:
        return False

    if last < first * 2.0:
        return False

    growth_steps = 0
    for i in range(1, len(history)):
        if history[i] > history[i - 1]:
            growth_steps += 1

    return growth_steps >= 4


def run_bleeding_detection(
    video_path: str,
    job_id: str,
    fps_sample: int = 5,
    conf: float = 0.25
) -> Dict[str, Any]:
    """
    Retorna:
      - events: lista de eventos
      - metrics: métricas agregadas
      - raw: lista bruta
      - output_video: caminho do vídeo anotado
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = max(int(video_fps / fps_sample), 1)

    model = load_model()
    if model is None:
        cap.release()
        return {
            "events": [],
            "metrics": {
                "note": "YOLO model not found. Train and place model.pt to enable bleeding detection."
            },
            "raw": [],
            "output_video": None,
        }

    storage_root = os.getenv("STORAGE_ROOT", "/app/storage")

    out_dir = os.path.join(storage_root, "results", job_id)
    os.makedirs(out_dir, exist_ok=True)

    output_video_path = os.path.join(out_dir, "bleeding_detected.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, float(video_fps), (w, h))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Não foi possível criar o vídeo de saída: {output_video_path}")

    frame_idx = 0
    raw = []

    gauze_soaked_count = 0
    gauze_total_count = 0

    blood_frames_detected = 0
    blood_large_frames = 0
    blood_max_ratio = 0.0
    pooling_frames = 0

    blood_ratio_history = deque(maxlen=6)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / video_fps

        results = model.predict(frame, imgsz=640, conf=conf, verbose=False)
        r = results[0]

        annotated = r.plot()

        out.write(annotated)

        detections = []
        blood_boxes = []

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())

                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    continue

                label = CLASS_NAMES[cls_id]

                detections.append(
                    {
                        "label": label,
                        "confidence": round(score, 4),
                    }
                )

                if label in ("Gauze", "StainedGauze", "RedGauze"):
                    gauze_total_count += 1
                    if label == "RedGauze":
                        gauze_soaked_count += 1

                if label == "Blood":
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    blood_boxes.append((x1, y1, x2, y2))

        blood_ratio = 0.0
        blood_level = None
        is_pooling = False

        if len(blood_boxes) > 0:
            blood_area = compute_total_bbox_area(blood_boxes)
            blood_ratio = blood_area / float(w * h)

            blood_max_ratio = max(blood_max_ratio, blood_ratio)

            blood_level = classify_blood_level(blood_ratio)
            blood_frames_detected += 1

            if blood_level == "blood_large":
                blood_large_frames += 1

        blood_ratio_history.append(blood_ratio)

        is_pooling = detect_pooling(blood_ratio_history)
        if is_pooling:
            pooling_frames += 1

        if detections or blood_ratio > 0:
            raw.append(
                {
                    "time": round(timestamp_sec, 2),
                    "detections": detections,
                    "blood_ratio": round(blood_ratio, 6),
                    "blood_level": blood_level,
                    "pooling": bool(is_pooling),
                    "blood_boxes": len(blood_boxes),
                }
            )

        frame_idx += 1

    cap.release()
    out.release()

    events = []

    if gauze_soaked_count >= 3:
        events.append(
            {
                "type": "gauze_soaked_high",
                "score": min(1.0, gauze_soaked_count / 10.0),
                "description": "Detecção frequente de RedGauze (gaze muito manchada). Revisão recomendada.",
            }
        )

    if blood_large_frames >= 3:
        events.append(
            {
                "type": "blood_large_frequent",
                "score": min(1.0, blood_large_frames / 10.0),
                "description": "Sangramento classificado como 'large' em vários frames. Revisão recomendada.",
            }
        )

    if pooling_frames >= 2:
        events.append(
            {
                "type": "blood_pooling_suspected",
                "score": min(1.0, pooling_frames / 6.0),
                "description": "Possível pooling: aumento consistente da área de sangue ao longo do tempo.",
            }
        )

    metrics = {
        "sample_fps": fps_sample,

        "gauze_total_detections": gauze_total_count,
        "gauze_soaked_frames": gauze_soaked_count,

        "blood_frames_detected": blood_frames_detected,
        "blood_large_frames": blood_large_frames,
        "blood_max_ratio": round(blood_max_ratio, 6),
        "pooling_frames": pooling_frames,

        "blood_thresholds": {
            "small_lt": 0.01,
            "medium_lt": 0.05
        }
    }

    return {
        "events": events,
        "metrics": metrics,
        "raw": raw,
        "output_video": output_video_path,
    }
