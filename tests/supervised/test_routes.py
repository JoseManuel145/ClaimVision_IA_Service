import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.supervised.presentation.routes import router
from app.modules.supervised.presentation.dependencies import (
    get_v2_predict_use_case,
    get_v2_retrain_use_case,
    get_v2_history_use_case,
    get_classifier,
)
from app.modules.supervised.domain.models import V2Prediction, RetrainJob


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v2")
    return app


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def mock_predict_use_case(now):
    m = AsyncMock()

    async def execute_side(image_bytes, filename):
        return V2Prediction(
            id="pred-1", filename=filename, class_id=2,
            tipo_dano="Rayadura", severidad="Bajo", confianza=0.92,
            prob_dist=[0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0],
            created_at=now,
        )

    m.execute.side_effect = execute_side
    return m


@pytest.fixture
def mock_retrain_use_case(now):
    m = AsyncMock()

    async def start_side(labels, files, epochs=40, lr=0.001):
        return RetrainJob(
            id="job-1", status="pending", total_epochs=epochs,
            current_epoch=0, best_accuracy=0.0,
            loss_history=[], error=None,
            created_at=now, completed_at=None,
        )

    m.start_retrain.side_effect = start_side
    m.get_job_status = AsyncMock(return_value=RetrainJob(
        id="job-1", status="completed", total_epochs=10,
        current_epoch=10, best_accuracy=0.95,
        loss_history=[0.5, 0.3, 0.1], error=None,
        created_at=now, completed_at=now,
    ))
    return m


@pytest.fixture
def mock_history_use_case(now):
    m = AsyncMock()

    async def execute_side(page=1, limit=20):
        return {
            "data": [
                {
                    "id": "pred-1", "filename": "test.jpg", "class_id": 2,
                    "tipo_dano": "Rayadura", "severidad": "Bajo", "confianza": 0.92,
                    "prob_dist": [0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0],
                    "created_at": now.isoformat(),
                }
            ],
            "total": 1, "page": page, "limit": limit,
        }

    m.execute.side_effect = execute_side
    return m


@pytest.fixture
def mock_classifier_service():
    m = MagicMock()
    m._device = "cpu"
    m.get_class_names.return_value = ["a", "b", "c", "d", "e", "f", "g"]
    return m


@pytest.fixture
def client(app, mock_predict_use_case, mock_retrain_use_case, mock_history_use_case, mock_classifier_service):
    app.dependency_overrides[get_v2_predict_use_case] = lambda: mock_predict_use_case
    app.dependency_overrides[get_v2_retrain_use_case] = lambda: mock_retrain_use_case
    app.dependency_overrides[get_v2_history_use_case] = lambda: mock_history_use_case
    app.dependency_overrides[get_classifier] = lambda: mock_classifier_service
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestPredictRoute:
    def test_predict_success(self, client, mock_predict_use_case):
        resp = client.post(
            "/api/v2/predict",
            files={"file": ("test.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "pred-1"
        assert data["tipo_dano"] == "Rayadura"
        assert data["class_id"] == 2
        mock_predict_use_case.execute.assert_awaited_once()

    def test_predict_rejects_non_image(self, client):
        resp = client.post(
            "/api/v2/predict",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_rejects_oversized(self, client):
        resp = client.post(
            "/api/v2/predict",
            files={"file": ("big.jpg", b"x" * (11 * 1024 * 1024), "image/jpeg")},
        )
        assert resp.status_code == 400


class TestRetrainRoute:
    def test_retrain_success(self, client, mock_retrain_use_case):
        labels = json.dumps([{"filename": "a.jpg", "label": 0}])
        resp = client.post(
            "/api/v2/retrain",
            data={"labels": labels, "epochs": "10", "lr": "0.001"},
            files=[("files", ("a.jpg", b"img", "image/jpeg")) for _ in range(7)],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        mock_retrain_use_case.start_retrain.assert_awaited_once()

    def test_retrain_invalid_labels_json(self, client):
        resp = client.post(
            "/api/v2/retrain",
            data={"labels": "not-json"},
            files=[("files", ("a.jpg", b"img", "image/jpeg")) for _ in range(7)],
        )
        assert resp.status_code == 400

    def test_retrain_fewer_than_7_images(self, client):
        labels = json.dumps([{"filename": "a.jpg", "label": 0}])
        resp = client.post(
            "/api/v2/retrain",
            data={"labels": labels},
            files=[("files", ("a.jpg", b"img", "image/jpeg")) for _ in range(3)],
        )
        assert resp.status_code == 400

    def test_retrain_status_success(self, client, mock_retrain_use_case):
        resp = client.get("/api/v2/retrain/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "completed"
        assert data["best_accuracy"] == 0.95
        mock_retrain_use_case.get_job_status.assert_awaited_once_with("job-1")

    def test_retrain_status_not_found(self, client, mock_retrain_use_case):
        mock_retrain_use_case.get_job_status.return_value = None
        resp = client.get("/api/v2/retrain/nonexistent")
        assert resp.status_code == 404


class TestHistoryRoute:
    def test_history_success(self, client):
        resp = client.get("/api/v2/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["tipo_dano"] == "Rayadura"


class TestHealthRoute:
    def test_health_success(self, client):
        resp = client.get("/api/v2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["num_classes"] == 7
