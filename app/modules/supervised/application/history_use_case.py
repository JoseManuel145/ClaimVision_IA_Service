from app.modules.supervised.domain.ports import V2PredictionRepository


class V2HistoryUseCase:
    def __init__(self, repository: V2PredictionRepository):
        self._repository = repository

    async def execute(self, page: int = 1, limit: int = 20) -> dict:
        items, total = await self._repository.list_paginated(page, limit)
        return {
            "data": [
                {
                    "id": i.id,
                    "filename": i.filename,
                    "class_id": i.class_id,
                    "tipo_dano": i.tipo_dano,
                    "severidad": i.severidad,
                    "confianza": i.confianza,
                    "prob_dist": i.prob_dist,
                    "created_at": i.created_at.isoformat(),
                }
                for i in items
            ],
            "total": total,
            "page": page,
            "limit": limit,
        }
