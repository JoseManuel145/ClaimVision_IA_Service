from pydantic import BaseModel
from typing import Optional


class DamageEntityResponse(BaseModel):
    tipo_dano: str
    severidad: str
    parte_afectada: str
    sintoma: str
    confianza: float


class TranscribirResponse(BaseModel):
    id: str
    filename: str
    texto: str
    duracion_seg: float
    entidades: list[DamageEntityResponse]
    created_at: str


class TranscribirJobResponse(BaseModel):
    job_id: str
    status: str
    progress: int


class TranscribirJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[TranscribirResponse] = None
    error: Optional[str] = None


class AnalizarRequest(BaseModel):
    texto: str


class AnalizarResponse(BaseModel):
    entidades: list[DamageEntityResponse]


class NlpHistoryItem(BaseModel):
    id: str
    filename: str
    texto: str
    duracion_seg: float
    entidades: list[DamageEntityResponse]
    created_at: str


class NlpHistoryResponse(BaseModel):
    data: list[NlpHistoryItem]
    total: int
    page: int
    limit: int
