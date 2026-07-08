from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from app.modules.nlp.presentation.schemas import (
    TranscribirResponse,
    AnalizarRequest,
    AnalizarResponse,
    NlpHistoryResponse,
    NlpHistoryItem,
    DamageEntityResponse,
)
from app.modules.nlp.presentation.dependencies import (
    get_transcribir_use_case,
    get_history_use_case,
    get_llm_service,
)
from app.modules.nlp.application.transcribir_use_case import TranscribirUseCase
from app.modules.nlp.application.history_use_case import HistoryUseCase
from app.modules.nlp.infra.llm.ollama_extractor import OllamaExtractor

router = APIRouter()


def _build_transcripcion_response(t) -> TranscribirResponse:
    return TranscribirResponse(
        id=t.id,
        filename=t.filename,
        texto=t.texto,
        duracion_seg=t.duracion_seg,
        entidades=[
            DamageEntityResponse(
                tipo_dano=e.tipo_dano,
                severidad=e.severidad,
                parte_afectada=e.parte_afectada,
                sintoma=e.sintoma,
                confianza=e.confianza,
            )
            for e in t.entidades
        ],
        created_at=t.created_at.isoformat(),
    )


@router.post(
    "/nlp/transcribir",
    response_model=TranscribirResponse,
    summary="Transcribir audio y extraer danos",
    description="Recibe un archivo de audio, lo transcribe con Whisper y extrae danos con LLM.",
)
async def transcribir(
    file: UploadFile = File(..., description="Archivo de audio"),
    use_case: TranscribirUseCase = Depends(get_transcribir_use_case),
):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un audio")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo esta vacio")
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El audio excede 25MB")

    transcripcion = await use_case.execute(contents, file.filename or "audio.m4a")
    return _build_transcripcion_response(transcripcion)


@router.post(
    "/nlp/analizar",
    response_model=AnalizarResponse,
    summary="Analizar texto directamente",
    description="Envia texto descriptivo de danos y recibe las entidades extraidas por el LLM.",
)
async def analizar(
    body: AnalizarRequest,
    llm: OllamaExtractor = Depends(get_llm_service),
):
    entidades = await llm.extraer_danos(body.texto)
    return AnalizarResponse(
        entidades=[
            DamageEntityResponse(
                tipo_dano=e.tipo_dano,
                severidad=e.severidad,
                parte_afectada=e.parte_afectada,
                sintoma=e.sintoma,
                confianza=e.confianza,
            )
            for e in entidades
        ],
    )


@router.get(
    "/nlp/history",
    response_model=NlpHistoryResponse,
    summary="Historial de transcripciones",
    description="Devuelve una lista paginada de todas las transcripciones con sus entidades.",
)
async def nlp_history(
    page: int = Query(1, ge=1, description="Numero de pagina"),
    limit: int = Query(20, ge=1, le=100, description="Elementos por pagina"),
    use_case: HistoryUseCase = Depends(get_history_use_case),
):
    items, total = await use_case.list_paginated(page, limit)
    return NlpHistoryResponse(
        data=[
            NlpHistoryItem(
                id=t.id,
                filename=t.filename,
                texto=t.texto,
                duracion_seg=t.duracion_seg,
                entidades=[
                    DamageEntityResponse(
                        tipo_dano=e.tipo_dano,
                        severidad=e.severidad,
                        parte_afectada=e.parte_afectada,
                        sintoma=e.sintoma,
                        confianza=e.confianza,
                    )
                    for e in t.entidades
                ],
                created_at=t.created_at.isoformat(),
            )
            for t in items
        ],
        total=total,
        page=page,
        limit=limit,
    )


@router.get(
    "/nlp/{id}",
    response_model=TranscribirResponse,
    summary="Detalle de una transcripcion",
    description="Obtiene una transcripcion y sus entidades por ID.",
)
async def nlp_detail(
    id: str,
    use_case: HistoryUseCase = Depends(get_history_use_case),
):
    transcripcion = await use_case.get_by_id(id)
    if transcripcion is None:
        raise HTTPException(status_code=404, detail="Transcripcion no encontrada")
    return _build_transcripcion_response(transcripcion)
