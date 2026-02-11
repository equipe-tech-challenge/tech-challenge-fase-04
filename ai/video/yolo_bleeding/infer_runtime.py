import os
from pathlib import Path
from typing import Dict, Any

import cv2
from ultralytics import YOLO


CLASS_NAMES = [
    "gauze_soaked",
    "gauze_stained",
    "gauze_clean"
]


def load_model() -> YOLO:
    model_path = Path("ai/video/yolo_bleeding/model.pt")
    if not model_path.exists():
        # Se não existir modelo treinado, retorna None e o pipeline segue vazio
        return None
    return YOLO(str(model_path))


def run_bleeding_detection(
    video_path: str,
    job_id: str,
    fps_sample: int = 5,
    conf: float = 0.25
) -> Dict[str, Any]:
    """
    Retorna:
      - events: lista de eventos (timestamps)
      - metrics: métricas agregadas
      - raw: lista bruta
      - output_video: caminho do vídeo anotado (se save_video=True)
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

    # -------------------------
    # 1) Preparar VideoWriter
    # -------------------------
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Se você quiser salvar o vídeo com boxes só nos frames amostrados:
        # mantém esse if como está.
        #
        # Se quiser salvar o vídeo COMPLETO (sem pular frames),
        # eu te mostro abaixo a versão.
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / video_fps

        results = model.predict(frame, imgsz=640, conf=conf, verbose=False)
        r = results[0]

        # -------------------------
        # 2) Desenhar boxes
        # -------------------------
        annotated = r.plot()

        # -------------------------
        # 3) Salvar frame no vídeo
        # -------------------------

        out.write(annotated)

        detections = []
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
                        "confidence": score,
                    }
                )

                if label.startswith("gauze"):
                    gauze_total_count += 1
                    if label == "gauze_soaked":
                        gauze_soaked_count += 1

        if detections:
            raw.append(
                {
                    "time": round(timestamp_sec, 2),
                    "detections": detections,
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
                "description": "Detecção frequente de gauze soaked. Revisão recomendada.",
            }
        )

    metrics = {
        "gauze_total_detections": gauze_total_count,
        "gauze_soaked_frames": gauze_soaked_count,
        "sample_fps": fps_sample,
    }

    return {
        "events": events,
        "metrics": metrics,
        "raw": raw,
        "output_video": output_video_path,
    }