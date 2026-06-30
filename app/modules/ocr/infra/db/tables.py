import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime
from app.core.database import Base


class OCRDocumentTable(Base):
    __tablename__ = "ocr_documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    text = Column(String, nullable=False)
    page_count = Column(Integer, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
