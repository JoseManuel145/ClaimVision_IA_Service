from app.modules.nlp.domain.models import VozTranscripcion
from app.modules.nlp.domain.ports import VozRepository


class HistoryUseCase:
    def __init__(self, repo: VozRepository):
        self._repo = repo

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[VozTranscripcion], int]:
        return await self._repo.list_paginated(page, limit)

    async def get_by_id(self, id: str) -> VozTranscripcion | None:
        return await self._repo.get_by_id(id)
