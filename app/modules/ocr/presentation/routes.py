from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from app.modules.ocr.presentation.schemas import (
    OcrResponse,
    OcrHistoryResponse,
    OcrHistoryItem,
)
from app.modules.ocr.presentation.dependencies import (
    get_ocr_use_case,
    get_ocr_repository,
)
from app.modules.ocr.application.ocr_use_case import OcrUseCase
from app.modules.ocr.infra.db.repository import PostgresOCRDocumentRepository

router = APIRouter()


@router.post(
    "/ocr",
    response_model=OcrResponse,
    summary="Extraer texto de un PDF",
    description="Recibe un PDF, extrae el texto con PyMuPDF + fallback OCR con Tesseract, y almacena el resultado.",
)
async def ocr(
    file: UploadFile = File(..., description="Documento PDF"),
    use_case: OcrUseCase = Depends(get_ocr_use_case),
):
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El PDF excede 10MB")

    doc = await use_case.execute(contents, file.filename or "unknown")
    return OcrResponse(
        id=doc.id,
        filename=doc.filename,
        text=doc.text,
        page_count=doc.page_count,
        created_at=doc.created_at.isoformat(),
    )


@router.get(
    "/ocr/history",
    response_model=OcrHistoryResponse,
    summary="Historial de documentos OCR",
    description="Devuelve una lista paginada de todos los documentos procesados por OCR.",
)
async def ocr_history(
    page: int = Query(1, ge=1, description="Número de página"),
    limit: int = Query(20, ge=1, le=100, description="Elementos por página"),
    repo: PostgresOCRDocumentRepository = Depends(get_ocr_repository),
):
    items, total = await repo.list_paginated(page, limit)
    return OcrHistoryResponse(
        data=[
            OcrHistoryItem(
                id=d.id,
                filename=d.filename,
                page_count=d.page_count,
                created_at=d.created_at.isoformat(),
            )
            for d in items
        ],
        total=total,
        page=page,
        limit=limit,
    )
