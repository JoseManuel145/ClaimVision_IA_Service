from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


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

    @staticmethod
    def create(
        filename: str,
        cluster_id: int,
        tipo_dano: str,
        severidad: str,
        confianza: float,
        distancia_centroide: float,
    ) -> "Inference":
        return Inference(
            id=str(uuid4()),
            filename=filename,
            cluster_id=cluster_id,
            tipo_dano=tipo_dano,
            severidad=severidad,
            confianza=confianza,
            distancia_centroide=distancia_centroide,
            created_at=datetime.now(timezone.utc),
        )


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
