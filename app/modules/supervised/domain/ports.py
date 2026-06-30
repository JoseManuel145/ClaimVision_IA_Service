from typing import Optional, Protocol
from app.modules.supervised.domain.models import V2Prediction, RetrainJob


class ClassifierService(Protocol):
    async def predict(self, tensor: object) -> tuple[int, float, list[float]]: ...

    def retrain(
        self,
        data_dir: str,
        epochs: int,
        lr: float,
        job_id: str,
        on_epoch_end: callable,
    ) -> None: ...

    def get_class_names(self) -> list[str]: ...

    def get_severity(self, confidence: float) -> str: ...


class V2PredictionRepository(Protocol):
    async def save(self, pred: V2Prediction) -> V2Prediction: ...

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[V2Prediction], int]: ...


class RetrainJobRepository(Protocol):
    async def save(self, job: RetrainJob) -> RetrainJob: ...

    async def get_by_id(self, job_id: str) -> Optional[RetrainJob]: ...

    async def update(self, job: RetrainJob) -> RetrainJob: ...
