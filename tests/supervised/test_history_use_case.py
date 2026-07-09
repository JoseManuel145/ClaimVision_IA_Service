import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from app.modules.supervised.application.history_use_case import V2HistoryUseCase
from app.modules.supervised.domain.models import V2Prediction


@pytest.mark.asyncio
async def test_v2_history_execute():
    now = datetime.now(timezone.utc)
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = (
        [
            V2Prediction(
                id="pred-1", filename="test.jpg", class_id=2,
                tipo_dano="Rayadura", severidad="Bajo", confianza=0.92,
                prob_dist=[0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0],
                created_at=now,
            )
        ],
        1,
    )

    use_case = V2HistoryUseCase(mock_repo)
    result = await use_case.execute(page=1, limit=20)

    assert result["total"] == 1
    assert result["page"] == 1
    assert len(result["data"]) == 1
    assert result["data"][0]["tipo_dano"] == "Rayadura"
    mock_repo.list_paginated.assert_awaited_once_with(1, 20)


@pytest.mark.asyncio
async def test_v2_history_empty():
    mock_repo = AsyncMock()
    mock_repo.list_paginated.return_value = ([], 0)

    use_case = V2HistoryUseCase(mock_repo)
    result = await use_case.execute(page=1, limit=20)

    assert result["total"] == 0
    assert len(result["data"]) == 0
