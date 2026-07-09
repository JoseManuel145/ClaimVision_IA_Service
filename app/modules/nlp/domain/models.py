from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DamageEntity:
    tipo_dano: str
    severidad: str
    parte_afectada: str
    sintoma: str
    confianza: float


@dataclass
class VozTranscripcion:
    id: str
    filename: str
    texto: str
    duracion_seg: float
    entidades: list[DamageEntity]
    created_at: datetime


@dataclass
class TranscripcionJob:
    id: str
    filename: str
    status: str
    progress: int
    result_id: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
