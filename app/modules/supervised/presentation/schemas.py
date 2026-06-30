from pydantic import BaseModel
from typing import Optional


class V2PredictResponse(BaseModel):
    id: str
    filename: str
    class_id: int
    tipo_dano: str
    severidad: str
    confianza: float
    prob_dist: list[float]
    created_at: str


class V2RetrainResponse(BaseModel):
    job_id: str
    status: str


class V2RetrainStatusResponse(BaseModel):
    job_id: str
    status: str
    total_epochs: int
    current_epoch: int
    best_accuracy: float
    loss_history: list[float]
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class V2HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    class_names: list[str]
