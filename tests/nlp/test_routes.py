import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.nlp.presentation.routes import router
from app.modules.nlp.presentation.dependencies import (
    get_transcripcion_job_use_case,
    get_history_use_case,
    get_llm_service,
)
from app.modules.nlp.domain.models import (
    VozTranscripcion, DamageEntity, TranscripcionJob,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def mock_job_use_case(now):
    m = MagicMock()

    async def start_job_side(audio_bytes, filename):
        return TranscripcionJob(
            id="job-1", filename=filename,
            status="pending", progress=0,
            result_id=None, error=None,
            created_at=now, updated_at=now,
        )

    m.start_job = AsyncMock(side_effect=start_job_side)

    m.get_job_status = AsyncMock(return_value=TranscripcionJob(
        id="job-1", filename="audio.m4a",
        status="completed", progress=100,
        result_id="trans-1", error=None,
        created_at=now, updated_at=now,
    ))
    return m


@pytest.fixture
def mock_llm():
    m = AsyncMock()
    m.extraer_danos.return_value = [
        DamageEntity(
            tipo_dano="golpe_puerta", severidad="Medio",
            parte_afectada="carroceria", sintoma="golpe en la puerta",
            confianza=0.85,
        )
    ]
    return m


@pytest.fixture
def mock_history_use_case(now):
    m = AsyncMock()

    async def get_by_id_side(id):
        if id != "trans-1":
            return None
        return VozTranscripcion(
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

    m.get_by_id = AsyncMock(side_effect=get_by_id_side)

    async def list_side(page=1, limit=20):
        return (
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

    m.list_paginated = AsyncMock(side_effect=list_side)
    return m


@pytest.fixture
def client(app, mock_job_use_case, mock_history_use_case, mock_llm):
    app.dependency_overrides[get_transcripcion_job_use_case] = lambda: mock_job_use_case
    app.dependency_overrides[get_history_use_case] = lambda: mock_history_use_case
    app.dependency_overrides[get_llm_service] = lambda: mock_llm
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestTranscribirRoute:
    def test_transcribir_returns_job(self, client, mock_job_use_case):
        resp = client.post(
            "/api/v1/nlp/transcribir",
            files={"file": ("audio.m4a", b"fake_audio", "audio/m4a")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "pending"
        assert data["progress"] == 0
        mock_job_use_case.start_job.assert_awaited_once()

    def test_transcribir_rejects_non_audio(self, client):
        resp = client.post(
            "/api/v1/nlp/transcribir",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert "audio" in resp.json()["detail"]

    def test_transcribir_rejects_empty(self, client):
        resp = client.post(
            "/api/v1/nlp/transcribir",
            files={"file": ("empty.m4a", b"", "audio/m4a")},
        )
        assert resp.status_code == 400

    def test_transcribir_rejects_oversized(self, client):
        resp = client.post(
            "/api/v1/nlp/transcribir",
            files={"file": ("big.m4a", b"x" * (26 * 1024 * 1024), "audio/m4a")},
        )
        assert resp.status_code == 400
        assert "25MB" in resp.json()["detail"]


class TestTranscribirStatusRoute:
    def test_status_returns_progress(self, client, mock_job_use_case):
        mock_job_use_case.get_job_status.return_value = TranscripcionJob(
            id="job-1", filename="audio.m4a",
            status="processing", progress=60,
            result_id=None, error=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        resp = client.get("/api/v1/nlp/transcribir/status/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "processing"
        assert data["progress"] == 60
        assert data["result"] is None
        mock_job_use_case.get_job_status.assert_awaited_with("job-1")

    def test_status_returns_result_when_completed(self, client, mock_job_use_case, mock_history_use_case):
        resp = client.get("/api/v1/nlp/transcribir/status/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["result"] is not None
        assert data["result"]["texto"] == "El auto tiene un golpe en la puerta"
        assert len(data["result"]["entidades"]) == 1
        mock_history_use_case.get_by_id.assert_awaited_with("trans-1")

    def test_status_returns_404(self, client, mock_job_use_case):
        mock_job_use_case.get_job_status.return_value = None
        resp = client.get("/api/v1/nlp/transcribir/status/nonexistent")
        assert resp.status_code == 404


class TestAnalizarRoute:
    def test_analizar_success(self, client):
        resp = client.post(
            "/api/v1/nlp/analizar",
            json={"texto": "El auto tiene un golpe en la puerta"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entidades"]) == 1
        assert data["entidades"][0]["tipo_dano"] == "golpe_puerta"


class TestHistoryRoute:
    def test_history_success(self, client, mock_history_use_case):
        resp = client.get("/api/v1/nlp/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert "entidades" in data["data"][0]
        mock_history_use_case.list_paginated.assert_awaited_once()

    def test_history_with_pagination(self, client, mock_history_use_case):
        client.get("/api/v1/nlp/history?page=2&limit=5")
        mock_history_use_case.list_paginated.assert_awaited_with(2, 5)


class TestDetailRoute:
    def test_detail_success(self, client, mock_history_use_case):
        resp = client.get("/api/v1/nlp/trans-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "trans-1"
        assert data["texto"] == "El auto tiene un golpe en la puerta"
        mock_history_use_case.get_by_id.assert_awaited_with("trans-1")

    def test_detail_not_found(self, client, mock_history_use_case):
        mock_history_use_case.get_by_id.return_value = None
        resp = client.get("/api/v1/nlp/nonexistent")
        assert resp.status_code == 404
