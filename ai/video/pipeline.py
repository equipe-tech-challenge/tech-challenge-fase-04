import os

from typing import Dict, Any
from pathlib import Path

from ai.video.yolo_bleeding.infer_runtime import run_bleeding_detection
from ai.video.consultation.nonverbal_analyzer import NonVerbalAnalyzer
from ai.video.physiotherapy.physio_analyzer import PhysioAnalyzer

from api.schemas.job import JobTypeVideo

from ai.audio.extract_audio import extract_audio_from_video
from ai.audio.pipeline import process_audio


def process_video(video_path: str, job_id: str, job_request: JobTypeVideo) -> Dict[str, Any]:
    """
      1) Cirurgia: sangramento (YOLOv8 custom)
      2) Consulta: não-verbal (MediaPipe)
      3) Fisioterapia: movimento (MediaPipe)
      4) Áudio: extração + análise (Librosa)
    """
    result = {
        "modules": {
            "surgery_bleeding": None,
            "consultation_nonverbal": None,
            "physiotherapy_movement": None,
            "consultation_audio": None
        },
        "events": [],
        "metrics": {},
        "scores": {},
    }

    if job_request == JobTypeVideo.surgery:
        # ==========================================================
        # 1) Cirurgia: sangramento (YOLO)
        # ==========================================================
        bleeding = run_bleeding_detection(video_path=video_path, job_id=job_id, fps_sample=5, conf=0.25)

        result["modules"]["surgery_bleeding"] = {
            "metrics": bleeding.get("metrics", {}),
            "raw_detections": bleeding.get("raw", [])[:200],
        }

        result["events"].extend(bleeding.get("events", []))
        result["metrics"].update({f"surgery_{k}": v for k, v in bleeding.get("metrics", {}).items()})

        surgery_risk_score = 0.0
        for e in bleeding.get("events", []):
            if e["type"] == "gauze_soaked_high":
                surgery_risk_score = max(surgery_risk_score, float(e.get("score", 0.0)))
            
            if e["type"] == "blood_large_frequent":
                surgery_risk_score = max(surgery_risk_score, float(e.get("score", 0.0)))

            if e["type"] == "blood_pooling_suspected":
                surgery_risk_score = max(surgery_risk_score, float(e.get("score", 0.0)))

        result["scores"]["Score Geral de Atenção"] = round(surgery_risk_score, 3)

    elif job_request == JobTypeVideo.consultation:
        # ==========================================================
        # 2) Consulta: não-verbal (MediaPipe)
        # ==========================================================
        nonverbal = NonVerbalAnalyzer().analyze_video(video_path=video_path, job_id=job_id, fps_sample=5)

        result["modules"]["consultation_nonverbal"] = nonverbal
        result["events"].extend(nonverbal.get("events", []))
        result["metrics"].update({f"consult_{k}": v for k, v in nonverbal.get("metrics", {}).items()})
        result["scores"].update({f"consult_{k}": v for k, v in nonverbal.get("scores", {}).items()})

        discomfort = float(result["scores"].get("consult_nonverbal_discomfort_score", 0.0))
        agitation = float(result["scores"].get("consult_agitation_score", 0.0))
        avoidance = float(result["scores"].get("consult_avoidance_score", 0.0))

        overall = max(discomfort, agitation, avoidance)
        result["scores"]["Score Geral de Atenção"] = round(overall, 3)

    elif job_request == JobTypeVideo.physiotherapy:
        # ==========================================================
        # 3) Fisioterapia: movimento (MediaPipe)
        # ==========================================================
        physio = PhysioAnalyzer().analyze_video(video_path=video_path, job_id=job_id, fps_sample=5)

        result["modules"]["physiotherapy_movement"] = physio
        result["events"].extend(physio.get("events", []))
        result["metrics"].update({f"physio_{k}": v for k, v in physio.get("metrics", {}).items()})
        result["scores"].update({f"physio_{k}": v for k, v in physio.get("scores", {}).items()})

        exec_symmetry = float(result["scores"].get("physio_symmetry_score", 0.0))
        exec_quality = float(result["scores"].get("physio_execution_quality_score", 0.0))

        overall = max(exec_symmetry, exec_quality)
        result["scores"]["Score Geral de Atenção"] = round(overall, 3)   
    
    elif job_request == JobTypeVideo.audio_analysis:
        # ==========================================================
        # 4) Áudio: extração + análise (Librosa)
        # ==========================================================
        work_dir = Path("storage/uploads")
        wav_path = str(work_dir / f"{job_id}.wav")

        try:
            extract_audio_from_video(video_path=video_path, out_wav_path=wav_path, sr=16000)
            audio_res = process_audio(wav_path=wav_path)

            result["modules"]["consultation_audio"] = audio_res
            result["events"].extend(audio_res.get("events", []))
            result["metrics"].update({f"audio_{k}": v for k, v in audio_res.get("metrics", {}).items()})
            result["scores"].update({f"audio_{k}": v for k, v in audio_res.get("scores", {}).items()})

            audio_hes = float(result["scores"].get("audio_hesitation_score", 0.0))
            audio_anx = float(result["scores"].get("audio_anxiety_vocal_score", 0.0))
            audio_appr = float(result["scores"].get("audio_postpartum_risk_proxy_score", 0.0))
            audio_trauma = float(result["scores"].get("audio_trauma_vocal_proxy_score", 0.0))

            overall = max(audio_hes, audio_anx, audio_appr, audio_trauma)
            result["scores"]["Score Geral de Atenção"] = round(overall, 3)

        except Exception as ex:
            result["modules"]["consultation_audio"] = {
                "metrics": {},
                "scores": {},
                "events": [{
                    "type": "audio_processing_failed",
                    "score": 0.0,
                    "description": f"Falha ao processar áudio: {str(ex)[:180]}"
                }]
            }

    if os.path.exists(video_path):
        os.remove(video_path)

    return result
