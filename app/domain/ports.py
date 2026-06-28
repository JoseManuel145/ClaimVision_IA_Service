from abc import ABC, abstractmethod
from typing import Optional
from app.domain.models import Inference, PredictionResult


class InferenceRepository(ABC):
    @abstractmethod
    async def save(self, inference: Inference) -> Inference: ...

    @abstractmethod
    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[Inference], int]: ...


class EncoderService(ABC):
    @abstractmethod
    async def encode(self, image_bytes: bytes) -> list[float]: ...


class ClusteringService(ABC):
    @abstractmethod
    async def predict(self, vector: list[float]) -> tuple[int, float]: ...

    @abstractmethod
    async def retrain(
        self, vectors: list[list[float]], k: int
    ) -> tuple[object, dict]: ...


class ClusterMapper(ABC):
    @abstractmethod
    def map(self, cluster_id: int, distance: float) -> PredictionResult: ...

    @abstractmethod
    def update_mapping(self, mapping: list[dict]) -> None: ...


class ImagePreprocessor(ABC):
    @abstractmethod
    async def preprocess(self, image_bytes: bytes) -> list[float]: ...
