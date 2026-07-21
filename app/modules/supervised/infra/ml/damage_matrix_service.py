import json
from pathlib import Path


class DamageMatrixService:
    def __init__(self, models_dir: str):
        self._matrix_path = Path(models_dir) / "damage_matrix.json"
        self._moneda = "MXN"
        self._costos: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not self._matrix_path.exists():
            return
        with open(self._matrix_path) as f:
            data = json.load(f)
        self._moneda = data.get("moneda", "MXN")
        self._costos = data.get("costos", {})

    def get_costo(self, tipo_dano: str, severidad: str) -> float:
        tipos = self._costos.get(tipo_dano)
        if tipos is None:
            return 0.0
        return tipos.get(severidad, 0.0)

    @property
    def moneda(self) -> str:
        return self._moneda
