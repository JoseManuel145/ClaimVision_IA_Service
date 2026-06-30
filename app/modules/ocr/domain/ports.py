from typing import Protocol
from app.modules.ocr.domain.models import OCRDocument


class OCRService(Protocol):
    async def extract(self, pdf_bytes: bytes) -> str: ...


class OCRDocumentRepository(Protocol):
    async def save(self, doc: OCRDocument) -> OCRDocument: ...

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[OCRDocument], int]: ...
