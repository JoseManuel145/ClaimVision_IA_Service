from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session

from app.modules.ocr.infra.db.repository import PostgresOCRDocumentRepository
from app.modules.ocr.infra.ocr.pymupdf_ocr_service import PyMuPDFOCRService
from app.modules.ocr.infra.ocr.tesseract_image_service import TesseractImageOCRService
from app.modules.ocr.infra.regex.regex_document_extractor import RegexDocumentExtractor
from app.modules.ocr.infra.validation.image_validator import ImageValidator
from app.modules.ocr.application.ocr_use_case import OcrUseCase
from app.modules.ocr.application.extract_poliza_use_case import ExtractPolizaUseCase
from app.modules.ocr.application.extract_ine_use_case import ExtractIneUseCase
from app.modules.ocr.application.extract_and_validate_use_case import ExtractAndValidateUseCase


def get_ocr_repository(session: AsyncSession = Depends(get_session)) -> PostgresOCRDocumentRepository:
    return PostgresOCRDocumentRepository(session)


def get_ocr_service() -> PyMuPDFOCRService:
    return PyMuPDFOCRService()


def get_image_ocr_service() -> TesseractImageOCRService:
    return TesseractImageOCRService()


def get_document_extractor() -> RegexDocumentExtractor:
    return RegexDocumentExtractor()


def get_image_validator() -> ImageValidator:
    return ImageValidator()


def get_ocr_use_case(
    repo: PostgresOCRDocumentRepository = Depends(get_ocr_repository),
    ocr: PyMuPDFOCRService = Depends(get_ocr_service),
) -> OcrUseCase:
    return OcrUseCase(repo, ocr)


def get_extract_poliza_use_case(
    repo: PostgresOCRDocumentRepository = Depends(get_ocr_repository),
    ocr: PyMuPDFOCRService = Depends(get_ocr_service),
    extractor: RegexDocumentExtractor = Depends(get_document_extractor),
) -> ExtractPolizaUseCase:
    return ExtractPolizaUseCase(ocr, extractor, repo)


def get_extract_ine_use_case(
    repo: PostgresOCRDocumentRepository = Depends(get_ocr_repository),
    ocr: PyMuPDFOCRService = Depends(get_ocr_service),
    image_ocr: TesseractImageOCRService = Depends(get_image_ocr_service),
    extractor: RegexDocumentExtractor = Depends(get_document_extractor),
) -> ExtractIneUseCase:
    return ExtractIneUseCase(ocr, image_ocr, extractor, repo)


def get_extract_and_validate_use_case(
    extract_poliza: ExtractPolizaUseCase = Depends(get_extract_poliza_use_case),
    extract_ine: ExtractIneUseCase = Depends(get_extract_ine_use_case),
) -> ExtractAndValidateUseCase:
    return ExtractAndValidateUseCase(extract_poliza, extract_ine)
