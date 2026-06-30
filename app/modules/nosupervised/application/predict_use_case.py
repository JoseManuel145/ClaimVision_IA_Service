from uuid import uuid4
from datetime import datetime, timezone

from app.modules.nosupervised.domain.models import Inference, PredictionResult
from app.modules.nosupervised.domain.ports import (
    InferenceRepository,
    ImagePreprocessor,
    EncoderService,
    ClusteringService,
    ClusterMapper,
)


class PredictUseCase:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        encoder: EncoderService,
        clustering: ClusteringService,
        mapper: ClusterMapper,
        repository: InferenceRepository,
    ):
        self._preprocessor = preprocessor
        self._encoder = encoder
        self._clustering = clustering
        self._mapper = mapper
        self._repository = repository

    async def execute(self, image_bytes: bytes, filename: str) -> Inference:
        tensor = await self._preprocessor.preprocess(image_bytes)
        vector = await self._encoder.encode(tensor)
        cluster_id, distance = await self._clustering.predict(vector)
        result: PredictionResult = self._mapper.map(cluster_id, distance)

        inference = Inference(
            id=str(uuid4()),
            filename=filename,
            cluster_id=result.cluster_id,
            tipo_dano=result.tipo_dano,
            severidad=result.severidad,
            confianza=result.confianza,
            distancia_centroide=result.distancia_centroide,
            created_at=datetime.now(timezone.utc),
        )
        await self._repository.save(inference)
        return inference
