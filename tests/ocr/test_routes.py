import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.ocr.presentation.routes import router
from app.modules.ocr.presentation.dependencies import (
    get_ocr_use_case,
    get_ocr_repository,
)
from app.modules.ocr.domain.models import OCRDocument


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def mock_ocr_use_case(now):
    m = AsyncMock()

    async def execute_side(pdf_bytes, filename):
        return OCRDocument(
            id="doc-1",
            filename=filename,
            text="Texto extraído del PDF",
            page_count=3,
            created_at=now,
        )

    m.execute.side_effect = execute_side
    return m


@pytest.fixture
def mock_ocr_repo(now):
    m = AsyncMock()
    m.list_paginated.return_value = (
        [
            OCRDocument(
                id="doc-1", filename="test.pdf", text="Texto extraído del PDF",
                page_count=3, created_at=now,
            )
        ],
        1,
    )
    return m


@pytest.fixture
def client(app, mock_ocr_use_case, mock_ocr_repo):
    app.dependency_overrides[get_ocr_use_case] = lambda: mock_ocr_use_case
    app.dependency_overrides[get_ocr_repository] = lambda: mock_ocr_repo
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestOcrRoute:
    def test_ocr_success(self, client, mock_ocr_use_case):
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("test.pdf", b"fake_pdf", "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "doc-1"
        assert data["text"] == "Texto extraído del PDF"
        assert data["page_count"] == 3
        mock_ocr_use_case.execute.assert_awaited_once()

    def test_ocr_rejects_non_pdf(self, client):
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400
        assert "PDF" in resp.json()["detail"]

    def test_ocr_rejects_empty_file(self, client):
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_ocr_rejects_oversized_file(self, client):
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("big.pdf", b"x" * (11 * 1024 * 1024), "application/pdf")},
        )
        assert resp.status_code == 400
        assert "10MB" in resp.json()["detail"]


class TestOcrHistoryRoute:
    def test_history_success(self, client):
        resp = client.get("/api/v1/ocr/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["data"]) == 1
        assert data["data"][0]["filename"] == "test.pdf"

    def test_history_with_pagination(self, client):
        resp = client.get("/api/v1/ocr/history?page=2&limit=5")
        assert resp.status_code == 200
