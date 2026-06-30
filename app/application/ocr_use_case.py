from app.domain.models import OCRDocument
from app.domain.ports import OCRDocumentRepository, OCRService


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
        doc = OCRDocument.create(filename=filename, text=text, page_count=0)
        saved = await self._repo.save(doc)
        return saved
