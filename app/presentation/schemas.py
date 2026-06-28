from pydantic import BaseModel, Field
from typing import Optional


class PredictResponse(BaseModel):
    id: str
    filename: str
    tipo_dano: str
    severidad: str
    confianza: float
    distancia_centroide: float
    created_at: str


class HistoryItem(BaseModel):
    id: str
    filename: str
    cluster_id: int
    tipo_dano: str
    severidad: str
    confianza: float
    distancia_centroide: float
    created_at: str


class HistoryResponse(BaseModel):
    data: list[HistoryItem]
    total: int
    page: int
    limit: int


class ErrorResponse(BaseModel):
    detail: str


class RetrainResponse(BaseModel):
    k: int
    silhouette: float
    davies_bouldin: float
    inertia: float
    mapping: list[dict]
    trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    k_value: Optional[int] = None
