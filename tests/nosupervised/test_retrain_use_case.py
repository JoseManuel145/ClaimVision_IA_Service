import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from app.modules.nosupervised.application.retrain_use_case import RetrainUseCase
from app.modules.nosupervised.domain.models import TrainingMetrics


@pytest.mark.asyncio
async def test_retrain_success(tmp_path):
    models_dir = str(tmp_path)
    mapping_file = tmp_path / "cluster_mapping.json"
    mapping_file.write_text('{"version": 0, "k": 3, "clusters": [], "silhouette": 0.0, "trained_at": ""}')

    images = [(b"img1", "a.jpg"), (b"img2", "b.jpg"), (b"img3", "c.jpg")]
    k = 3

    with (
        patch("app.modules.nosupervised.application.retrain_use_case.TorchImagePreprocessor") as MockPreprocessor,
        patch("app.modules.nosupervised.application.retrain_use_case.TorchEncoderService") as MockEncoder,
        patch("app.modules.nosupervised.application.retrain_use_case.SklearnClusteringService") as MockClustering,
        patch("app.modules.nosupervised.application.retrain_use_case.pickle.dump"),
    ):
        mock_prep = MockPreprocessor.return_value
        mock_prep.preprocess = AsyncMock(return_value="tensor")

        mock_enc = MockEncoder.return_value
        mock_enc.encode = AsyncMock(return_value=[0.1] * 128)

        mock_clust = MockClustering.return_value
        mock_km = MagicMock()
        mock_km.cluster_centers_ = np.array([[0.1] * 128 for _ in range(k)])
        mock_km.predict.return_value = np.array([0, 1, 2])
        mock_clust.retrain = AsyncMock(return_value=(mock_km, {
            "silhouette": 0.6,
            "davies_bouldin": 0.4,
            "inertia": 42.0,
            "ari": 0.5,
            "nmi": 0.7,
        }))

        use_case = RetrainUseCase(models_dir=models_dir)
        result = await use_case.execute(images, k)

    assert isinstance(result, TrainingMetrics)
    assert result.k == k
    assert result.silhouette == 0.6
    assert result.davies_bouldin == 0.4
    assert result.inertia == 42.0
    assert result.ari == 0.0
    assert result.nmi == 0.0
    assert len(result.mapping) == k

    updated_mapping = tmp_path / "cluster_mapping.json"
    import json
    mapping_data = json.loads(updated_mapping.read_text())
    assert mapping_data["version"] == 1
    assert mapping_data["k"] == k


@pytest.mark.asyncio
async def test_retrain_less_images_than_k(tmp_path):
    models_dir = str(tmp_path)

    with (
        patch("app.modules.nosupervised.application.retrain_use_case.TorchEncoderService"),
        patch("app.modules.nosupervised.application.retrain_use_case.TorchImagePreprocessor"),
    ):
        images = [(b"img1", "a.jpg"), (b"img2", "b.jpg")]
        k = 3

        use_case = RetrainUseCase(models_dir=models_dir)
        with pytest.raises(Exception):
            await use_case.execute(images, k)
