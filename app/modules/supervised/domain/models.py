from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class V2Prediction:
    id: str
    filename: str
    class_id: int
    tipo_dano: str
    severidad: str
    confianza: float
    prob_dist: list[float]
    created_at: datetime


@dataclass
class RetrainJob:
    id: str
    status: str
    total_epochs: int
    current_epoch: int
    best_accuracy: float
    loss_history: list[float]
    error: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


@dataclass
class PredictionItem:
    filename: str
    phash: str
    tipo_dano: str
    severidad: str
    confianza: float
    duplicado_de: Optional[str] = None


@dataclass
class PredictAllResult:
    predicciones: list[PredictionItem]
    total_imagenes: int
    imagenes_unicas: int
    duplicados_detectados: int


@dataclass
class DanoItem:
    tipo: str
    severidad: str
    costo_reparacion: float


@dataclass
class ResumenResult:
    precio_total: float
    danos: list[DanoItem]
    moneda: str
