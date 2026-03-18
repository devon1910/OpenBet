from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.sql import func

from src.database import Base


class JobStatus(Base):
    __tablename__ = "job_status"

    action = Column(String(50), primary_key=True)
    status = Column(String(20), nullable=False, default="idle")
    message = Column(Text, default="")
    extra_json = Column(Text, default="")  # JSON-encoded extra fields
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
