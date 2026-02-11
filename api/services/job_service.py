import json
from sqlalchemy.orm import Session
from database.models import Job


def get_job(db: Session, job_id: str) -> Job | None:
    return db.query(Job).filter(Job.id == job_id).first()


def job_to_response(job: Job) -> dict:
    result = None
    if job.result_json:
        try:
            result = json.loads(job.result_json)
        except Exception:
            result = None

    return {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "result": result,
        "report_available": bool(job.report_path),
        "error_message": job.error_message,
    }
