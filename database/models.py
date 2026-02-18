from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func
from database.db import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    job_type = Column(String, nullable=False)
    status = Column(String, nullable=False)

    input_path = Column(String, nullable=False)

    result_json = Column(Text, nullable=True)
    report_path = Column(String, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)
