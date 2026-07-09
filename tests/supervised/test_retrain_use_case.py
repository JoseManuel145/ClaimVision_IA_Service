import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.modules.supervised.application.retrain_use_case import V2RetrainUseCase
from app.modules.supervised.domain.models import RetrainJob


@pytest.mark.asyncio
async def test_start_retrain_creates_job_and_returns(mock_classifier, mock_retrain_job_repo):
    use_case = V2RetrainUseCase(mock_classifier, mock_retrain_job_repo)
    labels = [{"filename": "a.jpg", "label": 0}]
    files = [("a.jpg", b"imgdata")]

    with patch("asyncio.get_event_loop") as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        job = await use_case.start_retrain(labels, files, epochs=10, lr=0.001)

    assert job.status == "pending" or job.status == "training"
    assert job.total_epochs == 10
    mock_retrain_job_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_job_status_returns_job(mock_classifier, mock_retrain_job_repo):
    use_case = V2RetrainUseCase(mock_classifier, mock_retrain_job_repo)
    job = await use_case.get_job_status("job-1")

    assert job is not None
    assert job.id == "job-1"
    assert job.status == "completed"
    assert job.best_accuracy == 0.95
    mock_retrain_job_repo.get_by_id.assert_awaited_once_with("job-1")


@pytest.mark.asyncio
async def test_get_job_status_not_found(mock_classifier, mock_retrain_job_repo):
    mock_retrain_job_repo.get_by_id.return_value = None
    use_case = V2RetrainUseCase(mock_classifier, mock_retrain_job_repo)
    job = await use_case.get_job_status("nonexistent")

    assert job is None
