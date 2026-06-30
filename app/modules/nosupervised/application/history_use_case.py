from app.modules.nosupervised.domain.models import PaginatedResult
from app.modules.nosupervised.domain.ports import InferenceRepository


class HistoryUseCase:
    def __init__(self, repository: InferenceRepository):
        self._repository = repository

    async def execute(self, page: int = 1, limit: int = 20) -> PaginatedResult:
        items, total = await self._repository.list_paginated(page, limit)
        return PaginatedResult(
            data=[{
                "id": i.id,
                "filename": i.filename,
                "cluster_id": i.cluster_id,
                "tipo_dano": i.tipo_dano,
                "severidad": i.severidad,
                "confianza": i.confianza,
                "distancia_centroide": i.distancia_centroide,
                "created_at": i.created_at.isoformat(),
            } for i in items],
            total=total,
            page=page,
            limit=limit,
        )
