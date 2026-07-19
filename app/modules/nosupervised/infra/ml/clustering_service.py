import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from app.modules.nosupervised.domain.ports import ClusteringService


class SklearnClusteringService(ClusteringService):
    def __init__(self, models_dir: str):
        self._is_loaded = False
        self._kmeans = None
        model_path = Path(models_dir) / "kmeans.pkl"
        try:
            with open(model_path, "rb") as f:
                self._kmeans = pickle.load(f)
            self._is_loaded = True
        except Exception:
            pass

    async def predict(self, vector: list[float]) -> tuple[int, float]:
        if not self._is_loaded:
            raise RuntimeError("KMeans no fue cargado correctamente")
        dtype = self._kmeans.cluster_centers_.dtype
        arr = np.array([vector], dtype=dtype)
        cluster_id = int(self._kmeans.predict(arr)[0])
        distances = self._kmeans.transform(arr)[0]
        distance = float(distances[cluster_id])
        return cluster_id, distance

    async def retrain(
        self, vectors: list[list[float]], k: int
    ) -> tuple[KMeans, dict]:
        dtype = self._kmeans.cluster_centers_.dtype if hasattr(self._kmeans, 'cluster_centers_') else np.float64
        X = np.array(vectors, dtype=dtype)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        metrics = {}
        if k > 1:
            metrics["silhouette"] = float(silhouette_score(X, labels))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        metrics["inertia"] = float(km.inertia_)
        metrics["k"] = k

        return km, metrics
