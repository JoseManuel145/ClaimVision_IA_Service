import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from app.core.database import Base


class NlpTranscripcionTable(Base):
    __tablename__ = "nlp_transcripciones"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    texto = Column(Text, nullable=False)
    duracion_seg = Column(Float, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class NlpDamageEntityTable(Base):
    __tablename__ = "nlp_damage_entities"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transcripcion_id = Column(String, ForeignKey("nlp_transcripciones.id"), nullable=False)
    tipo_dano = Column(String(100), nullable=False)
    severidad = Column(String(10), nullable=False)
    parte_afectada = Column(String(100), nullable=False, default="")
    sintoma = Column(String(255), nullable=False, default="")
    confianza = Column(Float, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class NlpJobTable(Base):
    __tablename__ = "nlp_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    progress = Column(Integer, nullable=False, default=0)
    result_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
