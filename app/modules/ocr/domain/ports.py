from typing import Protocol
from app.modules.ocr.domain.models import OCRDocument, PolizaData, IneData


class OCRService(Protocol):
    async def extract(self, pdf_bytes: bytes) -> str: ...


class ImageOCRService(Protocol):
    async def extract_from_image(self, image_bytes: bytes, filename: str) -> str: ...


class DocumentStructuredExtractor(Protocol):
    async def extract_poliza(self, text: str) -> PolizaData: ...
    async def extract_ine(self, text: str) -> IneData: ...


class OCRDocumentRepository(Protocol):
    async def save(self, doc: OCRDocument) -> OCRDocument: ...

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[OCRDocument], int]: ...
