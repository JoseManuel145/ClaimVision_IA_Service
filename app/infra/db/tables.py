import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime
from app.database import Base


class InferenceTable(Base):
    __tablename__ = "inferences"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    cluster_id = Column(Integer, nullable=False)
    tipo_dano = Column(String(50), nullable=False)
    severidad = Column(String(10), nullable=False)
    confianza = Column(Float, nullable=False)
    distancia_centroide = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
