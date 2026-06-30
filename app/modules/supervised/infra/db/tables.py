import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text
from app.core.database import Base


class V2PredictionTable(Base):
    __tablename__ = "v2_predictions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    class_id = Column(Integer, nullable=False)
    tipo_dano = Column(String(50), nullable=False)
    severidad = Column(String(10), nullable=False)
    confianza = Column(Float, nullable=False)
    prob_dist = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class V2RetrainJobTable(Base):
    __tablename__ = "v2_retrain_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), nullable=False, default="pending")
    total_epochs = Column(Integer, nullable=False)
    current_epoch = Column(Integer, nullable=False, default=0)
    best_accuracy = Column(Float, nullable=False, default=0.0)
    loss_history = Column(JSON, nullable=False, default=list)
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
