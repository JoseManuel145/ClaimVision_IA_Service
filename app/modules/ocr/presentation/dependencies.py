from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session
from app.modules.ocr.infra.db.repository import PostgresOCRDocumentRepository
from app.modules.ocr.infra.ocr.pymupdf_ocr_service import PyMuPDFOCRService
from app.modules.ocr.application.ocr_use_case import OcrUseCase


def get_ocr_repository(session: AsyncSession = Depends(get_session)) -> PostgresOCRDocumentRepository:
    return PostgresOCRDocumentRepository(session)


def get_ocr_service() -> PyMuPDFOCRService:
    return PyMuPDFOCRService()


def get_ocr_use_case(
    repo: PostgresOCRDocumentRepository = Depends(get_ocr_repository),
    ocr: PyMuPDFOCRService = Depends(get_ocr_service),
) -> OcrUseCase:
    return OcrUseCase(repo, ocr)
