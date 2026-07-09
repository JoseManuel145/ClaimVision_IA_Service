import pytest
from unittest.mock import AsyncMock

from app.modules.nlp.application.transcribir_use_case import TranscribirUseCase


@pytest.mark.asyncio
async def test_transcribir_execute_success(mock_stt_service, mock_llm_service, mock_voz_repository):
    use_case = TranscribirUseCase(mock_voz_repository, mock_stt_service, mock_llm_service)
    result = await use_case.execute(b"fake_audio", "audio.m4a")

    assert result.filename == "audio.m4a"
    assert result.texto == "El auto tiene un golpe en la puerta"
    assert result.duracion_seg == 4.2
    assert len(result.entidades) == 1
    assert result.entidades[0].tipo_dano == "golpe_puerta"
    mock_stt_service.transcribir.assert_awaited_once_with(b"fake_audio", "audio.m4a")
    mock_llm_service.extraer_danos.assert_awaited_once()
    mock_voz_repository.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribir_pipeline_order(mock_voz_repository):
    call_order = []
    mock_stt = AsyncMock()
    mock_stt.transcribir.side_effect = lambda b, f: (
        call_order.append("stt") or ("texto", 5.0)
    )
    mock_llm = AsyncMock()
    mock_llm.extraer_danos.side_effect = lambda t: (
        call_order.append("llm") or []
    )

    use_case = TranscribirUseCase(mock_voz_repository, mock_stt, mock_llm)
    await use_case.execute(b"audio", "test.m4a")

    assert call_order == ["stt", "llm"]


@pytest.mark.asyncio
async def test_transcribir_propagates_stt_failure(mock_llm_service, mock_voz_repository):
    mock_stt = AsyncMock()
    mock_stt.transcribir.side_effect = RuntimeError("STT failed")

    use_case = TranscribirUseCase(mock_voz_repository, mock_stt, mock_llm_service)
    with pytest.raises(RuntimeError, match="STT failed"):
        await use_case.execute(b"bad", "fail.m4a")


@pytest.mark.asyncio
async def test_transcribir_propagates_llm_failure(mock_stt_service, mock_voz_repository):
    mock_llm = AsyncMock()
    mock_llm.extraer_danos.side_effect = RuntimeError("LLM failed")

    use_case = TranscribirUseCase(mock_voz_repository, mock_stt_service, mock_llm)
    with pytest.raises(RuntimeError, match="LLM failed"):
        await use_case.execute(b"audio", "test.m4a")
