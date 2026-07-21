import json
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Query
from app.modules.supervised.presentation.schemas import (
    V2PredictResponse,
    V2RetrainResponse,
    V2RetrainStatusResponse,
    V2HealthResponse,
    V2HistoryResponse,
    PredictAllResponse,
    ResumenRequest,
    ResumenResponse,
)
from app.modules.supervised.presentation.dependencies import (
    get_v2_predict_use_case,
    get_v2_retrain_use_case,
    get_v2_history_use_case,
    get_v2_predict_all_use_case,
    get_v2_resumen_use_case,
    get_classifier,
)
from app.modules.supervised.application.predict_use_case import V2PredictUseCase
from app.modules.supervised.application.retrain_use_case import V2RetrainUseCase
from app.modules.supervised.application.history_use_case import V2HistoryUseCase
from app.modules.supervised.application.predict_all_use_case import PredictAllUseCase
from app.modules.supervised.application.resumen_use_case import ResumenUseCase
from app.modules.supervised.infra.ml.classifier_service import ResNetClassifierService

router = APIRouter(tags=["Supervised"])


@router.post(
    "/predict",
    response_model=V2PredictResponse,
    summary="[v2] Clasificar daño vehicular",
    description="Usa modelo supervisado (ResNet18) para clasificar el daño en 7 clases incluyendo Sin Daño.",
)
async def predict_v2(
    file: UploadFile = File(..., description="Imagen del vehículo (JPG/PNG)"),
    use_case: V2PredictUseCase = Depends(get_v2_predict_use_case),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="La imagen excede 10MB")

    pred = await use_case.execute(contents, file.filename or "unknown")
    return V2PredictResponse(
        id=pred.id,
        filename=pred.filename,
        class_id=pred.class_id,
        tipo_dano=pred.tipo_dano,
        severidad=pred.severidad,
        confianza=pred.confianza,
        prob_dist=pred.prob_dist,
        created_at=pred.created_at.isoformat(),
    )


@router.post(
    "/predict-all",
    response_model=PredictAllResponse,
    summary="[v2] Clasificar múltiples imágenes con deduplicación",
    description="Recibe N imágenes, detecta duplicados por pHash, clasifica las únicas y retorna todos los resultados.",
)
async def predict_all_v2(
    files: list[UploadFile] = File(..., description="Imágenes del vehículo (JPG/PNG)"),
    use_case: PredictAllUseCase = Depends(get_v2_predict_all_use_case),
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Se requiere al menos una imagen")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Máximo 50 imágenes por request")

    images: list[tuple[str, bytes]] = []
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename} no es una imagen")
        contents = await f.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail=f"{f.filename} está vacío")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{f.filename} excede 10MB")
        images.append((f.filename or "unknown", contents))

    result = await use_case.execute(images)
    return PredictAllResponse(
        predicciones=[
            {
                "filename": p.filename,
                "phash": p.phash,
                "tipo_dano": p.tipo_dano,
                "severidad": p.severidad,
                "confianza": p.confianza,
                "duplicado_de": p.duplicado_de,
            }
            for p in result.predicciones
        ],
        resumen={
            "total_imagenes": result.total_imagenes,
            "imagenes_unicas": result.imagenes_unicas,
            "duplicados_detectados": result.duplicados_detectados,
        },
    )


@router.post(
    "/obtener-resumen",
    response_model=ResumenResponse,
    summary="[v2] Obtener resumen de costos de reparación",
    description="Recibe una lista de daños con severidad y retorna el costo total estimado basado en la matriz de daños.",
)
async def obtener_resumen_v2(
    body: ResumenRequest,
    use_case: ResumenUseCase = Depends(get_v2_resumen_use_case),
):
    result = use_case.execute([{"tipo": d.tipo, "severidad": d.severidad} for d in body.danos])
    return ResumenResponse(
        precio_total=result.precio_total,
        danos=[{"tipo": d.tipo, "severidad": d.severidad, "costo_reparacion": d.costo_reparacion} for d in result.danos],
        moneda=result.moneda,
    )


@router.post(
    "/retrain",
    response_model=V2RetrainResponse,
    summary="[v2] Re-entrenar modelo supervisado",
    description="Inicia entrenamiento async del clasificador. Enviar labels como JSON + imágenes.",
)
async def retrain_v2(
    labels: str = Form(..., description='JSON: [{"filename":"x.jpg","label":3}, ...]'),
    files: list[UploadFile] = File(..., description="Imágenes del dataset"),
    epochs: int = Form(40, ge=1, le=100, description="Cantidad de épocas"),
    lr: float = Form(0.001, gt=0, description="Learning rate"),
    use_case: V2RetrainUseCase = Depends(get_v2_retrain_use_case),
):
    try:
        labels_data = json.loads(labels)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="labels no es un JSON válido")

    if not isinstance(labels_data, list):
        raise HTTPException(status_code=400, detail="labels debe ser una lista")

    if len(files) < 7:
        raise HTTPException(status_code=400, detail=f"Se requieren al menos 7 imágenes (una por clase), se recibieron {len(files)}")

    file_data = []
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename} no es una imagen")
        contents = await f.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail=f"{f.filename} está vacío")
        file_data.append((f.filename or "unknown", contents))

    job = await use_case.start_retrain(labels_data, file_data, epochs, lr)
    return V2RetrainResponse(job_id=job.id, status=job.status)


@router.get(
    "/retrain/{job_id}",
    response_model=V2RetrainStatusResponse,
    summary="[v2] Estado del re-entreno",
    description="Consulta el progreso del entrenamiento asíncrono.",
)
async def retrain_status(
    job_id: str,
    use_case: V2RetrainUseCase = Depends(get_v2_retrain_use_case),
):
    job = await use_case.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return V2RetrainStatusResponse(
        job_id=job.id,
        status=job.status,
        total_epochs=job.total_epochs,
        current_epoch=job.current_epoch,
        best_accuracy=job.best_accuracy,
        loss_history=job.loss_history,
        error=job.error,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get(
    "/history",
    response_model=V2HistoryResponse,
    summary="[v2] Historial de inferencias",
    description="Devuelve lista paginada de todas las clasificaciones realizadas.",
)
async def history_v2(
    page: int = Query(1, ge=1, description="Número de página"),
    limit: int = Query(20, ge=1, le=100, description="Elementos por página"),
    use_case: V2HistoryUseCase = Depends(get_v2_history_use_case),
):
    result = await use_case.execute(page, limit)
    return V2HistoryResponse(**result)


@router.get(
    "/health",
    response_model=V2HealthResponse,
    summary="[v2] Health check",
    description="Estado del clasificador supervisado.",
)
async def health_v2(
    classifier: ResNetClassifierService = Depends(get_classifier),
):
    classes = classifier.get_class_names()
    if len(classes) == 0:
        raise HTTPException(status_code=500, detail="Modelo supervisado roto o sin clases")
        
    return V2HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(classifier._device),
        num_classes=len(classes),
        class_names=classes,
    )
