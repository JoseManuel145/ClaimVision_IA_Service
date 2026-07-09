import pytest
from unittest.mock import AsyncMock, MagicMock
from app.modules.nosupervised.application.predict_use_case import PredictUseCase
from app.modules.nosupervised.domain.models import PredictionResult


@pytest.mark.asyncio
async def test_predict_success():
    mock_preprocessor = AsyncMock()
    mock_encoder = AsyncMock()
    mock_clustering = AsyncMock()
    mock_mapper = MagicMock()
    mock_repo = AsyncMock()

    mock_preprocessor.preprocess.return_value = "tensor"
    mock_encoder.encode.return_value = [0.1] * 128
    mock_clustering.predict.return_value = (0, 2.5)
    mock_mapper.map.return_value = PredictionResult(
        tipo_dano="Rayadura",
        severidad="Medio",
        confianza=0.85,
        cluster_id=0,
        distancia_centroide=2.5,
    )

    use_case = PredictUseCase(
        mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_repo
    )
    result = await use_case.execute(b"fake_image", "test.jpg")

    assert result.filename == "test.jpg"
    assert result.tipo_dano == "Rayadura"
    assert result.severidad == "Medio"
    assert result.confianza == 0.85
    assert mock_repo.save.called


@pytest.mark.asyncio
async def test_predict_empty_image():
    use_case = PredictUseCase(
        AsyncMock(), AsyncMock(), AsyncMock(), MagicMock(), AsyncMock()
    )
    with pytest.raises(Exception):
        await use_case.execute(b"", "empty.jpg")
