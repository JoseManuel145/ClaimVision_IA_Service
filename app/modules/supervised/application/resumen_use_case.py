from app.modules.supervised.domain.models import DanoItem, ResumenResult
from app.modules.supervised.infra.ml.damage_matrix_service import DamageMatrixService


class ResumenUseCase:
    def __init__(self, matrix: DamageMatrixService):
        self._matrix = matrix

    def execute(self, danos: list[dict[str, str]]) -> ResumenResult:
        items: list[DanoItem] = []
        total = 0.0

        for d in danos:
            tipo = d.get("tipo", "")
            severidad = d.get("severidad", "")
            costo = self._matrix.get_costo(tipo, severidad)
            items.append(DanoItem(tipo=tipo, severidad=severidad, costo_reparacion=costo))
            total += costo

        return ResumenResult(
            precio_total=round(total, 2),
            danos=items,
            moneda=self._matrix.moneda,
        )
