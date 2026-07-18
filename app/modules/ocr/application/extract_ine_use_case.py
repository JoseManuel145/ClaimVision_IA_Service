from dataclasses import asdict
from uuid import uuid4
from datetime import datetime, timezone

from app.modules.ocr.domain.models import OCRDocument, DocumentExtraction
from app.modules.ocr.domain.ports import (
    OCRService,
    ImageOCRService,
    DocumentStructuredExtractor,
    OCRDocumentRepository,
)


class ExtractIneUseCase:
    MIN_OCR_TEXT_LENGTH = 50

    def __init__(
        self,
        ocr: OCRService,
        image_ocr: ImageOCRService,
        extractor: DocumentStructuredExtractor,
        repo: OCRDocumentRepository,
    ):
        self._ocr = ocr
        self._image_ocr = image_ocr
        self._extractor = extractor
        self._repo = repo

    async def execute(
        self, file_bytes: bytes, filename: str, content_type: str
    ) -> DocumentExtraction:
        if content_type and content_type.startswith("image/"):
            raw_text = await self._image_ocr.extract_from_image(file_bytes, filename)
        else:
            raw_text = await self._ocr.extract(file_bytes)

        if len(raw_text.strip()) < self.MIN_OCR_TEXT_LENGTH:
            raw_text = f"[OCR insuficiente: solo {len(raw_text.strip())} caracteres extraidos]"

        ine = await self._extractor.extract_ine(raw_text)

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
            document_type="ine",
            raw_text=raw_text,
            extracted_data=asdict(ine),
            created_at=doc.created_at,
        )
