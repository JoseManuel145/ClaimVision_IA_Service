from dataclasses import dataclass, field
from datetime import datetime


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
