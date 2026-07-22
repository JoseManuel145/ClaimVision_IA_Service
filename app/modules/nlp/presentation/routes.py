from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from app.modules.nlp.presentation.schemas import (
    TranscribirResponse,
    TranscribirJobResponse,
    TranscribirJobStatusResponse,
    AnalizarRequest,
    AnalizarResponse,
    NlpHistoryResponse,
    NlpHistoryItem,
    DamageEntityResponse,
)
from app.modules.nlp.presentation.dependencies import (
    get_transcribir_use_case,
    get_transcripcion_job_use_case,
    get_history_use_case,
    get_llm_service,
)
from app.modules.nlp.application.transcribir_use_case import TranscribirUseCase
from app.modules.nlp.application.transcripcion_job_use_case import TranscripcionJobUseCase
from app.modules.nlp.application.history_use_case import HistoryUseCase
from app.modules.nlp.infra.llm.groq_extractor import GroqExtractor

router = APIRouter(tags=["NLP"])


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
    response_model=TranscribirJobResponse,
    summary="Transcribir audio y extraer danos (asincrono)",
    description="Recibe un archivo de audio, crea un job y procesa en segundo plano. Consultar estado con GET /nlp/transcribir/status/{job_id}.",
)
async def transcribir(
    file: UploadFile = File(..., description="Archivo de audio"),
    job_use_case: TranscripcionJobUseCase = Depends(get_transcripcion_job_use_case),
):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser un audio")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo esta vacio")
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="El audio excede 25MB")

    job = await job_use_case.start_job(contents, file.filename or "audio.m4a")
    return TranscribirJobResponse(job_id=job.id, status=job.status, progress=job.progress)


@router.get(
    "/nlp/transcribir/status/{job_id}",
    response_model=TranscribirJobStatusResponse,
    summary="Estado de transcripcion asincrona",
    description="Devuelve el progreso del job. Si ya completo, incluye el resultado.",
)
async def transcribir_status(
    job_id: str,
    job_use_case: TranscripcionJobUseCase = Depends(get_transcripcion_job_use_case),
    history_use_case: HistoryUseCase = Depends(get_history_use_case),
):
    job = await job_use_case.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    result = None
    if job.status == "completed" and job.result_id:
        transcripcion = await history_use_case.get_by_id(job.result_id)
        if transcripcion:
            result = _build_transcripcion_response(transcripcion)

    return TranscribirJobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        result=result,
        error=job.error,
    )


@router.post(
    "/nlp/analizar",
    response_model=AnalizarResponse,
    summary="Analizar texto directamente",
    description="Envia texto descriptivo de danos y recibe las entidades extraidas por el LLM.",
)
async def analizar(
    body: AnalizarRequest,
    llm: GroqExtractor = Depends(get_llm_service),
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
