import pytest
from unittest.mock import AsyncMock

from app.modules.nlp.application.history_use_case import HistoryUseCase
from app.modules.nlp.domain.models import VozTranscripcion, DamageEntity
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_history_list_paginated(mock_voz_repository):
    use_case = HistoryUseCase(mock_voz_repository)
    items, total = await use_case.list_paginated(1, 20)

    assert total == 1
    assert len(items) == 1
    assert items[0].filename == "audio.m4a"
    assert items[0].texto == "El auto tiene un golpe en la puerta"
    mock_voz_repository.list_paginated.assert_awaited_once_with(1, 20)


@pytest.mark.asyncio
async def test_history_list_paginated_empty():
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = ([], 0)

    use_case = HistoryUseCase(mock_repo)
    items, total = await use_case.list_paginated(1, 20)

    assert total == 0
    assert len(items) == 0


@pytest.mark.asyncio
async def test_history_get_by_id(mock_voz_repository):
    use_case = HistoryUseCase(mock_voz_repository)
    result = await use_case.get_by_id("trans-1")

    assert result is not None
    assert result.id == "trans-1"
    assert result.texto == "El auto tiene un golpe en la puerta"
    assert len(result.entidades) == 1
    mock_voz_repository.get_by_id.assert_awaited_once_with("trans-1")


@pytest.mark.asyncio
async def test_history_get_by_id_not_found():
    mock_repo = AsyncMock()
    mock_repo.get_by_id.return_value = None

    use_case = HistoryUseCase(mock_repo)
    result = await use_case.get_by_id("nonexistent")

    assert result is None
