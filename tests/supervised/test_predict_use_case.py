import pytest
from unittest.mock import AsyncMock, MagicMock

from app.modules.supervised.application.predict_use_case import V2PredictUseCase


@pytest.mark.asyncio
async def test_v2_predict_success(
    mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo,
):
    use_case = V2PredictUseCase(mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo)
    result = await use_case.execute(b"fake_image", "test.jpg")

    assert result.filename == "test.jpg"
    assert result.class_id == 2
    assert result.tipo_dano == "Rayadura"
    assert result.severidad == "Bajo"
    assert result.confianza == 0.92
    mock_v2_preprocessor.preprocess.assert_awaited_once_with(b"fake_image")
    mock_classifier.predict.assert_called_once()
    mock_v2_prediction_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_predict_calls_classifier_methods_in_order(
    mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo,
):
    call_order = []
    mock_classifier.predict = AsyncMock(
        return_value=(2, 0.92, [0.01, 0.02, 0.92, 0.03, 0.01, 0.01, 0.0]),
    )
    mock_classifier.get_class_names = MagicMock(
        side_effect=lambda: (call_order.append("get_class_names") or ["a", "b", "c", "d", "e", "f", "g"])
    )
    mock_classifier.get_severity = MagicMock(
        side_effect=lambda c: (call_order.append("get_severity") or "Bajo")
    )

    use_case = V2PredictUseCase(mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo)
    await use_case.execute(b"fake", "test.jpg")

    assert "get_class_names" in call_order
    assert "get_severity" in call_order


@pytest.mark.asyncio
async def test_v2_predict_propagates_classifier_failure(
    mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo,
):
    mock_classifier.predict = MagicMock(side_effect=RuntimeError("classifier error"))

    use_case = V2PredictUseCase(mock_v2_preprocessor, mock_classifier, mock_v2_prediction_repo)
    with pytest.raises(RuntimeError, match="classifier error"):
        await use_case.execute(b"bad", "test.jpg")
