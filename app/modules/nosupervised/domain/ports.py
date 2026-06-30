from typing import Protocol

from app.modules.nosupervised.domain.models import Inference, PredictionResult


class InferenceRepository(Protocol):
    async def save(self, inference: Inference) -> Inference: ...

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[Inference], int]: ...


class EncoderService(Protocol):
    async def encode(self, image_bytes: bytes) -> list[float]: ...


class ClusteringService(Protocol):
    async def predict(self, vector: list[float]) -> tuple[int, float]: ...

    async def retrain(
        self, vectors: list[list[float]], k: int
    ) -> tuple[object, dict]: ...


class ClusterMapper(Protocol):
    def map(self, cluster_id: int, distance: float) -> PredictionResult: ...

    def update_mapping(self, mapping: list[dict]) -> None: ...


class ImagePreprocessor(Protocol):
    async def preprocess(self, image_bytes: bytes) -> list[float]: ...
