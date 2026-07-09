import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from app.modules.nosupervised.application.history_use_case import HistoryUseCase
from app.modules.nosupervised.domain.models import Inference


@pytest.mark.asyncio
async def test_history_use_case_execute():
    now = datetime.now(timezone.utc)
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = (
        [
            Inference(
                id="inf-1", filename="test.jpg", cluster_id=0,
                tipo_dano="Rayadura", severidad="Medio", confianza=0.85,
                distancia_centroide=2.5, created_at=now,
            )
        ],
        1,
    )

    use_case = HistoryUseCase(mock_repo)
    result = await use_case.execute(page=1, limit=20)

    assert result.total == 1
    assert result.page == 1
    assert result.limit == 20
    assert len(result.data) == 1
    assert result.data[0]["id"] == "inf-1"
    assert result.data[0]["tipo_dano"] == "Rayadura"
    mock_repo.list_paginated.assert_awaited_once_with(1, 20)


@pytest.mark.asyncio
async def test_history_use_case_empty():
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = ([], 0)

    use_case = HistoryUseCase(mock_repo)
    result = await use_case.execute(page=1, limit=20)

    assert result.total == 0
    assert len(result.data) == 0


@pytest.mark.asyncio
async def test_history_use_case_pagination():
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = ([], 50)

    use_case = HistoryUseCase(mock_repo)
    result = await use_case.execute(page=3, limit=10)

    assert result.page == 3
    assert result.limit == 10
    assert result.total == 50
    mock_repo.list_paginated.assert_awaited_once_with(3, 10)
