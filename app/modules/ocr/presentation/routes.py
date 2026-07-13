from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from app.modules.ocr.presentation.schemas import (
    OcrResponse,
    OcrHistoryResponse,
    OcrHistoryItem,
    PolizaExtractedResponse,
    IneExtractedResponse,
    ExtractAndValidateResponse,
    ValidationResult,
)
from app.modules.ocr.presentation.dependencies import (
    get_ocr_use_case,
    get_ocr_repository,
    get_extract_poliza_use_case,
    get_extract_ine_use_case,
    get_extract_and_validate_use_case,
)
from app.modules.ocr.application.ocr_use_case import OcrUseCase
from app.modules.ocr.application.extract_poliza_use_case import ExtractPolizaUseCase
from app.modules.ocr.application.extract_ine_use_case import ExtractIneUseCase
from app.modules.ocr.application.extract_and_validate_use_case import ExtractAndValidateUseCase
from app.modules.ocr.infra.db.repository import PostgresOCRDocumentRepository

router = APIRouter(tags=["OCR"])


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


@router.post(
    "/ocr/extract-poliza",
    response_model=PolizaExtractedResponse,
    summary="Extraer datos estructurados de una poliza de seguro",
    description="Recibe un PDF de poliza, extrae texto con OCR y estructura los campos con LLM.",
)
async def extract_poliza(
    file: UploadFile = File(..., description="PDF de la poliza de seguro"),
    use_case: ExtractPolizaUseCase = Depends(get_extract_poliza_use_case),
):
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo esta vacio")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El PDF excede 10MB")

    result = await use_case.execute(contents, file.filename or "unknown")
    data = result.extracted_data
    return PolizaExtractedResponse(
        id=result.id,
        filename=result.filename,
        numero_poliza=data.get("numero_poliza", ""),
        aseguradora=data.get("aseguradora", ""),
        nombre_asegurado=data.get("nombre_asegurado", ""),
        vehiculo_marca=data.get("vehiculo_marca", ""),
        vehiculo_modelo=data.get("vehiculo_modelo", ""),
        vehiculo_anio=data.get("vehiculo_anio", 0),
        vehiculo_placas=data.get("vehiculo_placas", ""),
        vehiculo_vin=data.get("vehiculo_vin"),
        vehiculo_color=data.get("vehiculo_color"),
        vigencia_inicio=data.get("vigencia_inicio") or "",
        vigencia_fin=data.get("vigencia_fin") or "",
    )


@router.post(
    "/ocr/extract-ine",
    response_model=IneExtractedResponse,
    summary="Extraer datos estructurados de una credencial INE",
    description="Recibe una imagen o PDF de credencial INE, extrae texto con OCR y estructura los campos con LLM.",
)
async def extract_ine(
    file: UploadFile = File(..., description="Imagen o PDF de la credencial INE"),
    use_case: ExtractIneUseCase = Depends(get_extract_ine_use_case),
):
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (JPG/PNG) o un PDF",
        )
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo esta vacio")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El archivo excede 10MB")

    result = await use_case.execute(
        contents, file.filename or "unknown", file.content_type or ""
    )
    data = result.extracted_data
    return IneExtractedResponse(
        id=result.id,
        filename=result.filename,
        nombre_completo=data.get("nombre_completo", ""),
        curp=data.get("curp", ""),
        rfc=data.get("rfc"),
        fecha_nacimiento=data.get("fecha_nacimiento", ""),
        sexo=data.get("sexo", ""),
        domicilio=data.get("domicilio", ""),
        clave_elector=data.get("clave_elector", ""),
    )


@router.post(
    "/ocr/extract-and-validate",
    response_model=ExtractAndValidateResponse,
    summary="Extraer y validar poliza contra INE",
    description="Recibe una poliza PDF y una credencial INE (imagen/PDF), extrae ambas y valida cruzadamente.",
)
async def extract_and_validate(
    poliza: UploadFile = File(..., description="PDF de la poliza de seguro"),
    ine: UploadFile = File(..., description="Imagen o PDF de la credencial INE"),
    use_case: ExtractAndValidateUseCase = Depends(get_extract_and_validate_use_case),
):
    poliza_bytes = await poliza.read()
    if len(poliza_bytes) == 0:
        raise HTTPException(status_code=400, detail="La poliza esta vacia")
    if len(poliza_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="La poliza excede 10MB")
    if poliza.content_type and poliza.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="La poliza debe ser un PDF")

    ine_bytes = await ine.read()
    if len(ine_bytes) == 0:
        raise HTTPException(status_code=400, detail="La INE esta vacia")
    if len(ine_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="La INE excede 10MB")

    result = await use_case.execute(
        poliza_bytes,
        poliza.filename or "unknown",
        ine_bytes,
        ine.filename or "unknown",
        ine.content_type or "",
    )

    poliza_ext = result["poliza"]
    ine_ext = result["ine"]
    val = result["validation"]

    poliza_data = poliza_ext.extracted_data
    ine_data = ine_ext.extracted_data

    return ExtractAndValidateResponse(
        poliza=PolizaExtractedResponse(
            id=poliza_ext.id,
            filename=poliza_ext.filename,
            numero_poliza=poliza_data.get("numero_poliza", ""),
            aseguradora=poliza_data.get("aseguradora", ""),
            nombre_asegurado=poliza_data.get("nombre_asegurado", ""),
            vehiculo_marca=poliza_data.get("vehiculo_marca", ""),
            vehiculo_modelo=poliza_data.get("vehiculo_modelo", ""),
            vehiculo_anio=poliza_data.get("vehiculo_anio", 0),
            vehiculo_placas=poliza_data.get("vehiculo_placas", ""),
            vehiculo_vin=poliza_data.get("vehiculo_vin"),
            vehiculo_color=poliza_data.get("vehiculo_color"),
            vigencia_inicio=poliza_data.get("vigencia_inicio", ""),
            vigencia_fin=poliza_data.get("vigencia_fin", ""),
        ),
        ine=IneExtractedResponse(
            id=ine_ext.id,
            filename=ine_ext.filename,
            nombre_completo=ine_data.get("nombre_completo", ""),
            curp=ine_data.get("curp", ""),
            rfc=ine_data.get("rfc"),
            fecha_nacimiento=ine_data.get("fecha_nacimiento", ""),
            sexo=ine_data.get("sexo", ""),
            domicilio=ine_data.get("domicilio", ""),
            clave_elector=ine_data.get("clave_elector", ""),
        ),
        validation=ValidationResult(
            poliza_vs_ine_nombre_match=val["poliza_vs_ine_nombre_match"],
            curp_rfc_consistent=val["curp_rfc_consistent"],
            detalles=val["detalles"],
        ),
    )
