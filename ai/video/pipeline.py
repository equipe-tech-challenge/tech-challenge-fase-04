from typing import Dict, Any

from ai.video.yolo_bleeding.infer_runtime import run_bleeding_detection


def process_video(video_path: str, job_id: str) -> Dict[str, Any]:
    """
    Pipeline MVP:
      - Cirurgia: sangramento (YOLOv8 custom)
      - Consulta não-verbal (vai entrar na Entrega 3)
      - Fisio pose (vai entrar na Entrega 3)
    """
    result = {
        "modules": {
            "surgery_bleeding": None,
            "consultation_nonverbal": None,
            "physiotherapy_movement": None,
        },
        "events": [],
        "metrics": {},
        "scores": {},
    }

    bleeding = run_bleeding_detection(video_path=video_path, job_id=job_id, fps_sample=5, conf=0.25)

    result["modules"]["surgery_bleeding"] = {
        "metrics": bleeding.get("metrics", {}),
        "raw_detections": bleeding.get("raw", [])[:200],  # limita pra não explodir o JSON
    }

    # Eventos e métricas globais
    result["events"].extend(bleeding.get("events", []))
    result["metrics"].update(bleeding.get("metrics", {}))

    # Score geral simples (MVP)
    result["scores"]["surgery_risk_score"] = 0.0
    for e in bleeding.get("events", []):
        if e["type"] == "bleeding_high":
            result["scores"]["surgery_risk_score"] = max(result["scores"]["surgery_risk_score"], e["score"])

    return result
