import json
import math
from pathlib import Path
from app.modules.nosupervised.domain.models import PredictionResult
from app.modules.nosupervised.domain.ports import ClusterMapper


class JsonClusterMapper(ClusterMapper):
    def __init__(self, models_dir: str):
        path = Path(models_dir) / "cluster_mapping.json"
        with open(path) as f:
            self._data = json.load(f)
        self._cluster_map = {c["id"]: c for c in self._data["clusters"]}
        max_dist = max(c.get("max_distance", 10.0) for c in self._cluster_map.values())
        self._max_dist = max(1.0, max_dist)

    def map(self, cluster_id: int, distance: float) -> PredictionResult:
        cluster = self._cluster_map.get(
            cluster_id,
            {"id": cluster_id, "tipo_dano": "Desconocido", "severidad_base": "Bajo"},
        )
        ratio = min(distance / self._max_dist, 1.0)
        if ratio < 0.33:
            severidad = "Bajo"
        elif ratio < 0.66:
            severidad = "Medio"
        else:
            severidad = "Alto"
        confianza = max(0.0, 1.0 - ratio)

        return PredictionResult(
            tipo_dano=cluster["tipo_dano"],
            severidad=severidad,
            confianza=round(confianza, 4),
            cluster_id=cluster_id,
            distancia_centroide=round(distance, 4),
        )

    def update_mapping(self, mapping: list[dict]) -> None:
        self._cluster_map = {c["id"]: c for c in mapping}
