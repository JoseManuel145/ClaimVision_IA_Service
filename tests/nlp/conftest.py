from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Optional
import pytest

from app.modules.nlp.domain.models import VozTranscripcion, DamageEntity, TranscripcionJob


@pytest.fixture
def mock_stt_service():
    m = AsyncMock()
    m.transcribir.return_value = ("El auto tiene un golpe en la puerta", 4.2)
    return m


@pytest.fixture
def mock_llm_service():
    m = AsyncMock()
    m.extraer_danos.return_value = [
        DamageEntity(
            tipo_dano="golpe_puerta",
            severidad="Medio",
            parte_afectada="carroceria",
            sintoma="golpe en la puerta",
            confianza=0.85,
        )
    ]
    return m


@pytest.fixture
def mock_voz_repository():
    m = AsyncMock()
    now = datetime.now(timezone.utc)
    m.save.return_value = VozTranscripcion(
        id="trans-1",
        filename="audio.m4a",
        texto="El auto tiene un golpe en la puerta",
        duracion_seg=4.2,
        entidades=[
            DamageEntity(
                tipo_dano="golpe_puerta",
                severidad="Medio",
                parte_afectada="carroceria",
                sintoma="golpe en la puerta",
                confianza=0.85,
            )
        ],
        created_at=now,
    )
    m.get_by_id.return_value = VozTranscripcion(
        id="trans-1",
        filename="audio.m4a",
        texto="El auto tiene un golpe en la puerta",
        duracion_seg=4.2,
        entidades=[
            DamageEntity(
                tipo_dano="golpe_puerta",
                severidad="Medio",
                parte_afectada="carroceria",
                sintoma="golpe en la puerta",
                confianza=0.85,
            )
        ],
        created_at=now,
    )
    m.list_paginated.return_value = (
        [
            VozTranscripcion(
                id="trans-1", filename="audio.m4a",
                texto="El auto tiene un golpe en la puerta",
                duracion_seg=4.2,
                entidades=[
                    DamageEntity(
                        tipo_dano="golpe_puerta", severidad="Medio",
                        parte_afectada="carroceria", sintoma="golpe en la puerta",
                        confianza=0.85,
                    )
                ],
                created_at=now,
            )
        ],
        1,
    )
    return m


@pytest.fixture
def mock_job_repository():
    m = AsyncMock()
    now = datetime.now(timezone.utc)

    def make_job(
        job_id="job-1",
        status="completed",
        progress=100,
        result_id: Optional[str] = "trans-1",
        error: Optional[str] = None,
    ):
        return TranscripcionJob(
            id=job_id, filename="audio.m4a",
            status=status, progress=progress,
            result_id=result_id, error=error,
            created_at=now, updated_at=now,
        )

    m.save.return_value = make_job()
    m.get_by_id.return_value = make_job()
    return m


@pytest.fixture
def mock_session_factory(mock_voz_repository, mock_job_repository):
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.close = AsyncMock()

    async def async_enter():
        return mock_session

    async def async_exit(*args):
        pass

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=lambda: mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    mock_factory = MagicMock(return_value=mock_cm)
    return mock_factory
