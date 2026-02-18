from enum import Enum
from pydantic import BaseModel
from typing import Optional, Any, Dict

class JobTypeVideo(str, Enum):
    surgery = "Surgery: bleeding detection"
    consultation = "Consultation: non-verbal"
    physiotherapy = "Physiotherapy: movement"
    audio_analysis = "Video audio analysis: hesitation, anxiety and trauma"
    
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
