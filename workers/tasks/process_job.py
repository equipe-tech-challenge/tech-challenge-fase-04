import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from workers.celery_app import celery_app
from database.db import SessionLocal
from database.models import Job

from reports.pdf_report import generate_pdf_report

from ai.video.pipeline import process_video


@celery_app.task(name="process_job")
def process_job(job_id: str):
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
            video_result = process_video(job.input_path, job_id)

            result["modules"] = video_result.get("modules", {})
            result["events"] = video_result.get("events", [])
            result["metrics"] = video_result.get("metrics", {})
            result["scores"] = video_result.get("scores", {})

        elif job.job_type == "audio":
            result["metrics"]["note"] = "Audio pipeline not implemented yet (Entrega 4)."

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