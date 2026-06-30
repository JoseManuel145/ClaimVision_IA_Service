from dataclasses import dataclass
from datetime import datetime


@dataclass
class Inference:
    id: str
    filename: str
    cluster_id: int
    tipo_dano: str
    severidad: str
    confianza: float
    distancia_centroide: float
    created_at: datetime


@dataclass
class PredictionResult:
    tipo_dano: str
    severidad: str
    confianza: float
    cluster_id: int
    distancia_centroide: float


@dataclass
class TrainingMetrics:
    k: int
    silhouette: float
    davies_bouldin: float
    ari: float
    nmi: float
    inertia: float
    mapping: list[dict]
    trained_at: str


@dataclass
class PaginatedResult:
    data: list
    total: int
    page: int
    limit: int
