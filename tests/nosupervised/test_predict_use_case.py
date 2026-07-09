import pytest
from unittest.mock import AsyncMock, MagicMock

from app.modules.nosupervised.application.predict_use_case import PredictUseCase
from app.modules.nosupervised.domain.models import PredictionResult


@pytest.mark.asyncio
async def test_predict_success(
    mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
):
    use_case = PredictUseCase(
        mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
    )
    result = await use_case.execute(b"fake_image", "test.jpg")

    assert result.filename == "test.jpg"
    assert result.tipo_dano == "Rayadura"
    assert result.severidad == "Medio"
    assert result.confianza == 0.85
    mock_preprocessor.preprocess.assert_awaited_once_with(b"fake_image")
    mock_encoder.encode.assert_awaited_once()
    mock_clustering.predict.assert_awaited_once()
    mock_mapper.map.assert_called_once()
    mock_inference_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_predict_calls_all_steps_in_order(
    mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
):
    call_order = []

    mock_preprocessor.preprocess = AsyncMock(side_effect=lambda x: (call_order.append("preprocess") or "tensor"))
    mock_encoder.encode = AsyncMock(side_effect=lambda x: (call_order.append("encode") or [0.1] * 128))
    mock_clustering.predict = AsyncMock(side_effect=lambda x: (call_order.append("predict") or (0, 2.5)))

    use_case = PredictUseCase(
        mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
    )
    await use_case.execute(b"fake", "test.jpg")

    assert call_order == ["preprocess", "encode", "predict"]


@pytest.mark.asyncio
async def test_predict_propagates_preprocessor_failure(
    mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
):
    mock_preprocessor.preprocess = AsyncMock(side_effect=ValueError("invalid image"))

    use_case = PredictUseCase(
        mock_preprocessor, mock_encoder, mock_clustering, mock_mapper, mock_inference_repo,
    )
    with pytest.raises(ValueError, match="invalid image"):
        await use_case.execute(b"bad", "test.jpg")
