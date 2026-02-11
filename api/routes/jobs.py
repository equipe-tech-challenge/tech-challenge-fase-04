import os
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from sqlalchemy.orm import Session

from database.db import SessionLocal
from database.models import Job

from api.schemas.job import JobCreateResponse, JobStatusResponse
from api.services.storage_service import get_uploads_dir
from api.services.job_service import get_job, job_to_response

from workers.tasks.process_job import process_job

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/video", response_model=JobCreateResponse)
async def create_video_job(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido.")

    job_id = str(uuid.uuid4())
    uploads_dir = get_uploads_dir()

    ext = Path(file.filename).suffix.lower() or ".mp4"
    input_path = f"{uploads_dir}/{job_id}{ext}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    db: Session = SessionLocal()
    try:
        job = Job(
            id=job_id,
            job_type="video",
            status="queued",
            input_path=input_path
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    process_job.delay(job_id)

    return JobCreateResponse(job_id=job_id, status="queued")


@router.post("/audio", response_model=JobCreateResponse)
async def create_audio_job(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido.")

    job_id = str(uuid.uuid4())
    uploads_dir = get_uploads_dir()

    ext = Path(file.filename).suffix.lower() or ".wav"
    input_path = f"{uploads_dir}/{job_id}{ext}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    db: Session = SessionLocal()
    try:
        job = Job(
            id=job_id,
            job_type="audio",
            status="queued",
            input_path=input_path
        )
        db.add(job)
        db.commit()
    finally:
        db.close()

    process_job.delay(job_id)

    return JobCreateResponse(job_id=job_id, status="queued")


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    db: Session = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado.")

        data = job_to_response(job)
        return JobStatusResponse(**data)
    finally:
        db.close()


@router.get("/{job_id}/report")
def download_report(job_id: str):
    db: Session = SessionLocal()
    try:
        job = get_job(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job não encontrado.")

        if not job.report_path or not os.path.exists(job.report_path):
            raise HTTPException(status_code=404, detail="Relatório não disponível.")

        return FileResponse(
            job.report_path,
            media_type="application/pdf",
            filename=f"report_{job_id}.pdf"
        )
    finally:
        db.close()
