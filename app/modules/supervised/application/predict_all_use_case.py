from app.core.config import settings
from app.modules.supervised.domain.models import PredictionItem, PredictAllResult
from app.modules.supervised.domain.ports import ClassifierService
from app.modules.supervised.infra.ml.preprocessor import SupervisedPreprocessor
from app.modules.supervised.infra.ml.image_hasher import ImageHasher


class PredictAllUseCase:
    def __init__(
        self,
        preprocessor: SupervisedPreprocessor,
        classifier: ClassifierService,
        hasher: ImageHasher,
        confidence_threshold: float = settings.MIN_CONFIDENCE_THRESHOLD,
    ):
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._hasher = hasher
        self._confidence_threshold = confidence_threshold

    async def execute(
        self, images: list[tuple[str, bytes]]
    ) -> PredictAllResult:
        hashes: list[str] = []
        unique_indices: list[int] = []
        index_of_first: dict[str, int] = {}

        for filename, content in images:
            h = self._hasher.compute_hash(content)
            hashes.append(h)

            matched = None
            for existing_hash, existing_idx in zip(
                [hashes[i] for i in unique_indices], unique_indices
            ):
                if self._hasher.are_duplicates(h, existing_hash):
                    matched = existing_idx
                    break

            if matched is not None:
                index_of_first[len(hashes) - 1] = matched
            else:
                index_of_first[len(hashes) - 1] = len(hashes) - 1
                unique_indices.append(len(hashes) - 1)

        class_names = self._classifier.get_class_names()
        predictions_cache: dict[int, PredictionItem] = {}

        for idx in unique_indices:
            filename, content = images[idx]
            tensor = await self._preprocessor.preprocess(content)
            class_id, confidence, _ = await self._classifier.predict(tensor)

            if confidence < self._confidence_threshold:
                tipo_dano = "Sin Daño"
                severidad = "Ninguno"
            else:
                tipo_dano = class_names[class_id] if class_id < len(class_names) else "Desconocido"
                severidad = self._classifier.get_severity(confidence)

            predictions_cache[idx] = PredictionItem(
                filename=filename,
                phash=hashes[idx],
                tipo_dano=tipo_dano,
                severidad=severidad,
                confianza=round(confidence, 4),
            )

        result_items: list[PredictionItem] = []
        for i, (filename, _) in enumerate(images):
            first_idx = index_of_first[i]
            pred = predictions_cache[first_idx]
            item = PredictionItem(
                filename=filename,
                phash=pred.phash,
                tipo_dano=pred.tipo_dano,
                severidad=pred.severidad,
                confianza=pred.confianza,
                duplicado_de=images[first_idx][0] if first_idx != i else None,
            )
            result_items.append(item)

        dupes = sum(1 for item in result_items if item.duplicado_de is not None)

        return PredictAllResult(
            predicciones=result_items,
            total_imagenes=len(images),
            imagenes_unicas=len(unique_indices),
            duplicados_detectados=dupes,
        )
