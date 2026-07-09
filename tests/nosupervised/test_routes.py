import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from app.modules.nosupervised.presentation.routes import router
from app.modules.nosupervised.presentation.dependencies import (
    get_predict_use_case,
    get_history_use_case,
    get_retrain_use_case,
    get_encoder,
    get_clustering,
    get_mapper,
)
from app.modules.nosupervised.domain.models import Inference, PredictionResult


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def mock_services(now):
    from app.modules.nosupervised.domain.models import TrainingMetrics, PaginatedResult

    async def predict_side(image_bytes, filename):
        return Inference(
            id="inf-1",
            filename=filename or "unknown",
            cluster_id=0,
            tipo_dano="Rayadura",
            severidad="Medio",
            confianza=0.85,
            distancia_centroide=2.5,
            created_at=now,
        )

    mock_predict = MagicMock()
    mock_predict.execute = AsyncMock(side_effect=predict_side)

    async def history_side(page=1, limit=20):
        return PaginatedResult(
            data=[{
                "id": "inf-1", "filename": "test.jpg", "cluster_id": 0,
                "tipo_dano": "Rayadura", "severidad": "Medio", "confianza": 0.85,
                "distancia_centroide": 2.5, "created_at": now.isoformat(),
            }],
            total=1, page=page, limit=limit,
        )

    mock_history = MagicMock()
    mock_history.execute = AsyncMock(side_effect=history_side)

    mock_retrain = MagicMock()
    mock_retrain.execute = AsyncMock(return_value=TrainingMetrics(
        k=3, silhouette=0.6, davies_bouldin=0.4, ari=0.5, nmi=0.7,
        inertia=42.0, mapping=[{"id": 0}], trained_at=now.isoformat(),
    ))

    return {
        "predict": mock_predict,
        "history": mock_history,
        "retrain": mock_retrain,
    }


@pytest.fixture
def client(app, mock_services):
    app.dependency_overrides[get_predict_use_case] = lambda: mock_services["predict"]
    app.dependency_overrides[get_history_use_case] = lambda: mock_services["history"]
    app.dependency_overrides[get_retrain_use_case] = lambda: mock_services["retrain"]
    app.dependency_overrides[get_encoder] = lambda: MagicMock()
    app.dependency_overrides[get_clustering] = lambda: MagicMock()
    app.dependency_overrides[get_mapper] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestPredictRoute:
    def test_predict_success(self, client, mock_services):
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.jpg", b"fake_image", "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "inf-1"
        assert data["tipo_dano"] == "Rayadura"
        assert data["confianza"] == 0.85
        mock_services["predict"].execute.assert_awaited_once()

    def test_predict_rejects_non_image(self, client):
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert "imagen" in resp.json()["detail"]

    def test_predict_rejects_empty_file(self, client, mock_services):
        mock_services["predict"].side_effect = None
        mock_services["predict"].return_value = None
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_rejects_oversized_file(self, client):
        resp = client.post(
            "/api/v1/predict",
            files={"file": ("big.jpg", b"x" * (11 * 1024 * 1024), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "10MB" in resp.json()["detail"]


class TestHistoryRoute:
    def test_history_success(self, client):
        resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["tipo_dano"] == "Rayadura"

    def test_history_with_pagination(self, client):
        resp = client.get("/api/v1/history?page=2&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert data["limit"] == 5


class TestRetrainRoute:
    def test_retrain_success(self, client):
        resp = client.post(
            "/api/v1/retrain",
            data={"k": "3"},
            files=[
                ("files", ("a.jpg", b"img1", "image/jpeg")),
                ("files", ("b.jpg", b"img2", "image/jpeg")),
                ("files", ("c.jpg", b"img3", "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["k"] == 3
        assert data["silhouette"] == 0.6

    def test_retrain_fewer_images_than_k(self, client):
        resp = client.post(
            "/api/v1/retrain",
            data={"k": "5"},
            files=[
                ("files", ("a.jpg", b"img1", "image/jpeg")),
                ("files", ("b.jpg", b"img2", "image/jpeg")),
            ],
        )
        assert resp.status_code == 400
        assert "5 imágenes" in resp.json()["detail"]


class TestHealthRoute:
    def test_health_success(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
