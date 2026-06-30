import json
import io
import pickle
from datetime import datetime, timezone
from pathlib import Path
from app.modules.nosupervised.infra.ml.preprocessor import TorchImagePreprocessor
from app.modules.nosupervised.infra.ml.clustering_service import SklearnClusteringService
from app.modules.nosupervised.infra.ml.encoder_service import TorchEncoderService
from app.modules.nosupervised.domain.models import TrainingMetrics


class RetrainUseCase:
    def __init__(self, models_dir: str):
        self._models_dir = Path(models_dir)
        self._preprocessor = TorchImagePreprocessor()
        self._encoder = TorchEncoderService(models_dir)

    async def execute(
        self, images: list[tuple[bytes, str]], k: int
    ) -> TrainingMetrics:
        vectors = []
        for image_bytes, _fname in images:
            tensor = await self._preprocessor.preprocess(image_bytes)
            vec = await self._encoder.encode(tensor)
            vectors.append(vec)

        clustering = SklearnClusteringService(str(self._models_dir))
        km, metrics = await clustering.retrain(vectors, k)

        model_path = self._models_dir / "kmeans.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(km, f)

        mapping = self._build_mapping(km, vectors)
        mapping_path = self._models_dir / "cluster_mapping.json"
        with open(mapping_path) as f:
            current = json.load(f)
        current["version"] = current.get("version", 0) + 1
        current["k"] = k
        current["clusters"] = mapping
        current["silhouette"] = metrics.get("silhouette", 0.0)
        current["trained_at"] = datetime.now(timezone.utc).isoformat()
        with open(mapping_path, "w") as f:
            json.dump(current, f, indent=2)

        return TrainingMetrics(
            k=k,
            silhouette=metrics.get("silhouette", 0.0),
            davies_bouldin=metrics.get("davies_bouldin", 0.0),
            ari=0.0,
            nmi=0.0,
            inertia=metrics.get("inertia", 0.0),
            mapping=mapping,
            trained_at=current["trained_at"],
        )

    def _build_mapping(self, km, vectors) -> list[dict]:
        import numpy as np
        dtype = km.cluster_centers_.dtype
        X = np.array(vectors, dtype=dtype)
        labels = km.predict(X)
        mapping = []
        for i, center in enumerate(km.cluster_centers_):
            mask = labels == i
            count = int(mask.sum())
            avg_dist = float(np.linalg.norm(X[mask] - center, axis=1).mean()) if count > 0 else 0.0
            mapping.append({
                "id": int(i),
                "tipo_dano": f"Cluster_{i}",
                "severidad_base": "Medio",
                "count": count,
                "avg_distance": round(avg_dist, 4),
            })
        return mapping
