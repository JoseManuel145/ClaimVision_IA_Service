from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

from app.modules.ocr.domain.models import OCRDocument, DocumentExtraction
from app.modules.ocr.domain.ports import (
    OCRService,
    DocumentStructuredExtractor,
    OCRDocumentRepository,
)


class ExtractPolizaUseCase:
    def __init__(
        self,
        ocr: OCRService,
        extractor: DocumentStructuredExtractor,
        repo: OCRDocumentRepository,
    ):
        self._ocr = ocr
        self._extractor = extractor
        self._repo = repo

    async def execute(self, pdf_bytes: bytes, filename: str) -> DocumentExtraction:
        raw_text = await self._ocr.extract(pdf_bytes)
        poliza = await self._extractor.extract_poliza(raw_text)

        doc = OCRDocument(
            id=str(uuid4()),
            filename=filename,
            text=raw_text,
            page_count=0,
            created_at=datetime.now(timezone.utc),
        )
        await self._repo.save(doc)

        return DocumentExtraction(
            id=doc.id,
            filename=filename,
            document_type="poliza",
            raw_text=raw_text,
            extracted_data=asdict(poliza),
            created_at=doc.created_at,
        )
