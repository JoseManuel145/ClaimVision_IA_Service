from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import pytest

from app.modules.nosupervised.domain.models import Inference, PredictionResult, PaginatedResult


@pytest.fixture
def mock_preprocessor():
    m = AsyncMock()
    m.preprocess.return_value = [0.1] * 224
    return m


@pytest.fixture
def mock_encoder():
    m = AsyncMock()
    m.encode.return_value = [0.1] * 128
    return m


@pytest.fixture
def mock_clustering():
    m = AsyncMock()
    m.predict.return_value = (0, 2.5)
    return m


@pytest.fixture
def mock_mapper():
    m = MagicMock()
    m.map.return_value = PredictionResult(
        tipo_dano="Rayadura",
        severidad="Medio",
        confianza=0.85,
        cluster_id=0,
        distancia_centroide=2.5,
    )
    return m


@pytest.fixture
def mock_inference_repo():
    m = AsyncMock()
    now = datetime.now(timezone.utc)
    m.save.return_value = Inference(
        id="inf-1",
        filename="test.jpg",
        cluster_id=0,
        tipo_dano="Rayadura",
        severidad="Medio",
        confianza=0.85,
        distancia_centroide=2.5,
        created_at=now,
    )
    m.list_paginated.return_value = (
        [
            Inference(
                id="inf-1", filename="test.jpg", cluster_id=0,
                tipo_dano="Rayadura", severidad="Medio", confianza=0.85,
                distancia_centroide=2.5, created_at=now,
            )
        ],
        1,
    )
    return m
