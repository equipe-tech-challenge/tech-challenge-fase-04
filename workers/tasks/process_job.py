import json
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from api.schemas.job import JobTypeVideo
from workers.celery_app import celery_app
from database.db import SessionLocal
from database.models import Job

from reports.pdf_report import generate_pdf_report

from ai.video.pipeline import process_video

from ai.audio.pipeline import process_audio


@celery_app.task(name="process_job")
def process_job(job_id: str, job_request: JobTypeVideo = None):
    db: Session = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        db.commit()

        result = {
            "job_id": job.id,
            "job_type": job.job_type,
            "input_path": job.input_path,
            "events": [],
            "metrics": {},
            "scores": {},
            "modules": {},
        }

        if job.job_type == "video":
            video_result = process_video(job.input_path, job_id, job_request)

            result["modules"] = video_result.get("modules", {})
            result["events"] = video_result.get("events", [])
            result["metrics"] = video_result.get("metrics", {})
            result["scores"] = video_result.get("scores", {})

        elif job.job_type == "audio":          
            audio_res = process_audio(wav_path=job.input_path)
            
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

        pdf_path = generate_pdf_report(job, result)

        job.result_json = json.dumps(result, ensure_ascii=False)
        job.report_path = pdf_path
        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)

        db.commit()

    except Exception as ex:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "error"
            job.error_message = str(ex)
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()