from uuid import uuid4
from datetime import datetime, timezone

from app.modules.supervised.domain.models import V2Prediction
from app.modules.supervised.domain.ports import ClassifierService, V2PredictionRepository
from app.modules.supervised.infra.ml.preprocessor import SupervisedPreprocessor


class V2PredictUseCase:
    def __init__(
        self,
        preprocessor: SupervisedPreprocessor,
        classifier: ClassifierService,
        repository: V2PredictionRepository,
    ):
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._repository = repository

    async def execute(self, image_bytes: bytes, filename: str) -> V2Prediction:
        tensor = await self._preprocessor.preprocess(image_bytes)
        class_id, confidence, prob_dist = await self._classifier.predict(tensor)
        class_names = self._classifier.get_class_names()
        tipo_dano = class_names[class_id]
        severidad = self._classifier.get_severity(confidence)

        pred = V2Prediction(
            id=str(uuid4()),
            filename=filename,
            class_id=class_id,
            tipo_dano=tipo_dano,
            severidad=severidad,
            confianza=round(confidence, 4),
            prob_dist=[round(p, 4) for p in prob_dist],
            created_at=datetime.now(timezone.utc),
        )
        await self._repository.save(pred)
        return pred
