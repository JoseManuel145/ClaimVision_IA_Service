from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
import pytest

from app.modules.supervised.domain.models import V2Prediction, RetrainJob


@pytest.fixture
def mock_v2_preprocessor():
    m = AsyncMock()
    m.preprocess.return_value = "tensor"
    return m


@pytest.fixture
def mock_classifier():
    m = MagicMock()
    m.predict = AsyncMock(return_value=(2, 0.92, [0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0]))
    m.get_class_names.return_value = [
        "Sin Dano", "Golpe", "Rayadura", "Abolladura",
        "Roto", "Oxido", "Desperfecto",
    ]
    m.get_severity.return_value = "Bajo"
    return m


@pytest.fixture
def mock_v2_prediction_repo():
    m = AsyncMock()
    now = datetime.now(timezone.utc)
    m.save.return_value = V2Prediction(
        id="pred-1", filename="test.jpg", class_id=2,
        tipo_dano="Rayadura", severidad="Bajo", confianza=0.92,
        prob_dist=[0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0],
        created_at=now,
    )
    m.list_paginated.return_value = (
        [
            V2Prediction(
                id="pred-1", filename="test.jpg", class_id=2,
                tipo_dano="Rayadura", severidad="Bajo", confianza=0.92,
                prob_dist=[0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0],
                created_at=now,
            )
        ],
        1,
    )
    return m


@pytest.fixture
def mock_retrain_job_repo():
    m = AsyncMock()
    now = datetime.now(timezone.utc)
    m.save.return_value = RetrainJob(
        id="job-1", status="pending", total_epochs=10,
        current_epoch=0, best_accuracy=0.0,
        loss_history=[], error=None,
        created_at=now, completed_at=None,
    )
    m.get_by_id.return_value = RetrainJob(
        id="job-1", status="completed", total_epochs=10,
        current_epoch=10, best_accuracy=0.95,
        loss_history=[0.5, 0.3, 0.1], error=None,
        created_at=now, completed_at=now,
    )
    return m
