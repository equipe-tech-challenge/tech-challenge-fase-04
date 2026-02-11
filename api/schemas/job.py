from pydantic import BaseModel
from typing import Optional, Any, Dict


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    result: Optional[Dict[str, Any]] = None
    report_available: bool
    error_message: Optional[str] = None
