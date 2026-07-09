from unittest.mock import AsyncMock
from datetime import datetime, timezone
import pytest

from app.modules.ocr.domain.models import OCRDocument


@pytest.fixture
def mock_ocr_service():
    m = AsyncMock()
    m.extract.return_value = "Texto extraído del PDF"
    return m


@pytest.fixture
def mock_ocr_repo():
    m = AsyncMock()
    now = datetime.now(timezone.utc)
    m.save.return_value = OCRDocument(
        id="doc-1",
        filename="test.pdf",
        text="Texto extraído del PDF",
        page_count=3,
        created_at=now,
    )
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
