import json
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Query
from app.modules.supervised.presentation.schemas import (
    V2PredictResponse,
    V2RetrainResponse,
    V2RetrainStatusResponse,
    V2HealthResponse,
)
from app.modules.supervised.presentation.dependencies import (
    get_v2_predict_use_case,
    get_v2_retrain_use_case,
    get_classifier,
)
from app.modules.supervised.application.predict_use_case import V2PredictUseCase
from app.modules.supervised.application.retrain_use_case import V2RetrainUseCase
from app.modules.supervised.infra.ml.classifier_service import ResNetClassifierService

router = APIRouter()


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
    "/health",
    response_model=V2HealthResponse,
    summary="[v2] Health check",
    description="Estado del clasificador supervisado.",
)
async def health_v2(
    classifier: ResNetClassifierService = Depends(get_classifier),
):
    return V2HealthResponse(
        status="ok",
        model_loaded=True,
        device=str(classifier._device),
        num_classes=len(classifier.get_class_names()),
        class_names=classifier.get_class_names(),
    )
