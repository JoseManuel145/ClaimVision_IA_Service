import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.modules.nlp.application.transcripcion_job_use_case import TranscripcionJobUseCase
from app.modules.nlp.domain.models import TranscripcionJob


@pytest.mark.asyncio
async def test_start_job_creates_job_and_returns(mock_stt_service, mock_llm_service, mock_session_factory):
    use_case = TranscripcionJobUseCase(mock_stt_service, mock_llm_service, mock_session_factory)

    with patch("asyncio.create_task") as mock_create_task:
        mock_create_task.return_value = None
        job = await use_case.start_job(b"fake_audio", "audio.m4a")

    assert job.status == "pending"
    assert job.progress == 0
    assert job.filename == "audio.m4a"
    assert job.result_id is None
    assert job.error is None
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_get_job_status_returns_job(mock_stt_service, mock_llm_service, mock_session_factory, mock_job_repository):
    use_case = TranscripcionJobUseCase(mock_stt_service, mock_llm_service, mock_session_factory)

    use_case._session_factory = MagicMock()
    async def fake_session():
        class FakeCtx:
            def __init__(self):
                self.repo = mock_job_repository
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
        return FakeCtx()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock()
    cm.__aenter__.return_value = type('Sess', (), {'__aenter__': AsyncMock(return_value=AsyncMock()), '__aexit__': AsyncMock()})()

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=lambda: AsyncMock())
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    use_case._session_factory.return_value = mock_cm

    mock_session = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session

    from app.modules.nlp.infra.db.repository import PostgresTranscripcionJobRepository
    with patch.object(PostgresTranscripcionJobRepository, 'get_by_id', return_value=mock_job_repository.get_by_id.return_value):
        job = await use_case.get_job_status("job-1")

    assert job is not None
    assert job.id == "job-1"


@pytest.mark.asyncio
async def test_process_completes_successfully(mock_stt_service, mock_llm_service):
    now = datetime.now(timezone.utc)
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_factory = MagicMock(return_value=mock_cm)

    use_case = TranscripcionJobUseCase(mock_stt_service, mock_llm_service, mock_factory)

    mock_job_repo = AsyncMock()
    mock_job_repo.get_by_id.return_value = TranscripcionJob(
        id="job-1", filename="audio.m4a",
        status="processing", progress=10,
        result_id=None, error=None,
        created_at=now, updated_at=now,
    )

    mock_voz_repo = AsyncMock()
    mock_voz_repo.save.return_value.id = "trans-1"

    from app.modules.nlp.infra.db.repository import PostgresTranscripcionJobRepository, PostgresVozRepository
    with (
        patch.object(PostgresTranscripcionJobRepository, '__init__', return_value=None),
        patch.object(PostgresTranscripcionJobRepository, 'get_by_id', return_value=mock_job_repo.get_by_id.return_value),
        patch.object(PostgresTranscripcionJobRepository, 'update', return_value=mock_job_repo.get_by_id.return_value),
        patch.object(PostgresVozRepository, '__init__', return_value=None),
        patch.object(PostgresVozRepository, 'save', return_value=mock_voz_repo.save.return_value),
    ):
        await use_case._process("job-1", b"fake_audio", "audio.m4a")

    mock_stt_service.transcribir.assert_awaited_once()
    mock_llm_service.extraer_danos.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_marks_failed_on_stt_error(mock_stt_service, mock_llm_service):
    mock_stt_service.transcribir.side_effect = RuntimeError("STT crashed")

    now = datetime.now(timezone.utc)
    mock_session = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_factory = MagicMock(return_value=mock_cm)

    use_case = TranscripcionJobUseCase(mock_stt_service, mock_llm_service, mock_factory)

    mock_job_repo = AsyncMock()
    mock_job_repo.get_by_id.return_value = TranscripcionJob(
        id="job-1", filename="audio.m4a",
        status="processing", progress=10,
        result_id=None, error=None,
        created_at=now, updated_at=now,
    )

    from app.modules.nlp.infra.db.repository import PostgresTranscripcionJobRepository
    with (
        patch.object(PostgresTranscripcionJobRepository, '__init__', return_value=None),
        patch.object(PostgresTranscripcionJobRepository, 'get_by_id', return_value=mock_job_repo.get_by_id.return_value),
        patch.object(PostgresTranscripcionJobRepository, 'update'),
    ):
        await use_case._process("job-1", b"bad_audio", "fail.m4a")

    updated_job = mock_job_repo.get_by_id.return_value
    assert updated_job.status == "failed"
    assert updated_job.error is not None


@pytest.mark.asyncio
async def test_process_marks_failed_on_llm_error(mock_stt_service, mock_llm_service):
    mock_llm_service.extraer_danos.side_effect = RuntimeError("LLM crashed")

    now = datetime.now(timezone.utc)
    mock_session = AsyncMock()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_factory = MagicMock(return_value=mock_cm)

    use_case = TranscripcionJobUseCase(mock_stt_service, mock_llm_service, mock_factory)

    mock_job_repo = AsyncMock()
    mock_job_repo.get_by_id.return_value = TranscripcionJob(
        id="job-1", filename="audio.m4a",
        status="processing", progress=10,
        result_id=None, error=None,
        created_at=now, updated_at=now,
    )

    from app.modules.nlp.infra.db.repository import PostgresTranscripcionJobRepository
    with (
        patch.object(PostgresTranscripcionJobRepository, '__init__', return_value=None),
        patch.object(PostgresTranscripcionJobRepository, 'get_by_id', return_value=mock_job_repo.get_by_id.return_value),
        patch.object(PostgresTranscripcionJobRepository, 'update'),
    ):
        await use_case._process("job-1", b"audio", "test.m4a")

    updated_job = mock_job_repo.get_by_id.return_value
    assert updated_job.status == "failed"
