from uuid import uuid4
from datetime import datetime, timezone

from app.modules.ocr.domain.models import OCRDocument
from app.modules.ocr.domain.ports import OCRDocumentRepository, OCRService


class OcrUseCase:
    def __init__(
        self,
        repo: OCRDocumentRepository,
        ocr: OCRService,
    ):
        self._repo = repo
        self._ocr = ocr

    async def execute(self, pdf_bytes: bytes, filename: str) -> OCRDocument:
        text = await self._ocr.extract(pdf_bytes)
        doc = OCRDocument(
            id=str(uuid4()),
            filename=filename,
            text=text,
            page_count=0,
            created_at=datetime.now(timezone.utc),
        )
        saved = await self._repo.save(doc)
        return saved
