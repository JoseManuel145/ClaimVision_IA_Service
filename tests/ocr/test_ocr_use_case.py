import pytest
from unittest.mock import AsyncMock

from app.modules.ocr.application.ocr_use_case import OcrUseCase


@pytest.mark.asyncio
async def test_ocr_execute_success(mock_ocr_service, mock_ocr_repo):
    use_case = OcrUseCase(mock_ocr_repo, mock_ocr_service)
    result = await use_case.execute(b"fake_pdf", "test.pdf")

    assert result.filename == "test.pdf"
    assert result.text == "Texto extraído del PDF"
    mock_ocr_service.extract.assert_awaited_once_with(b"fake_pdf")
    mock_ocr_repo.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_ocr_execute_empty_pdf(mock_ocr_repo):
    mock_service = AsyncMock()
    mock_service.extract.return_value = ""
    mock_ocr_repo.save.return_value = None
    mock_ocr_repo.save.side_effect = lambda doc: doc

    use_case = OcrUseCase(mock_ocr_repo, mock_service)
    result = await use_case.execute(b"", "empty.pdf")

    assert result.text == ""
    mock_service.extract.assert_awaited_once_with(b"")


@pytest.mark.asyncio
async def test_ocr_execute_propagates_service_failure(mock_ocr_repo):
    mock_service = AsyncMock()
    mock_service.extract.side_effect = RuntimeError("OCR failed")

    use_case = OcrUseCase(mock_ocr_repo, mock_service)
    with pytest.raises(RuntimeError, match="OCR failed"):
        await use_case.execute(b"bad", "fail.pdf")
