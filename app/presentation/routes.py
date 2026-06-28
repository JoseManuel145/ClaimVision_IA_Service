from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Query
from app.presentation.schemas import (
    PredictResponse,
    HistoryResponse,
    RetrainResponse,
    HealthResponse,
    HistoryItem,
)
from app.presentation.dependencies import (
    get_predict_use_case,
    get_history_use_case,
    get_retrain_use_case,
    get_encoder,
    get_clustering,
    get_mapper,
)
from app.application.predict_use_case import PredictUseCase
from app.application.history_use_case import HistoryUseCase
from app.application.retrain_use_case import RetrainUseCase
from app.infra.ml.encoder_service import TorchEncoderService
from app.infra.ml.clustering_service import SklearnClusteringService
from app.infra.mapping.cluster_mapper import JsonClusterMapper

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predecir tipo de daño vehicular",
    description="Recibe una imagen de daño vehicular, la procesa a través del encoder y K-Means, y devuelve el tipo de daño, severidad y confianza.",
)
async def predict(
    file: UploadFile = File(..., description="Imagen del daño vehicular (JPG/PNG)"),
    use_case: PredictUseCase = Depends(get_predict_use_case),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="La imagen excede 10MB")

    inference = await use_case.execute(contents, file.filename or "unknown")
    return PredictResponse(
        id=inference.id,
        filename=inference.filename,
        tipo_dano=inference.tipo_dano,
        severidad=inference.severidad,
        confianza=inference.confianza,
        distancia_centroide=inference.distancia_centroide,
        created_at=inference.created_at.isoformat(),
    )


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Consultar historial de inferencias",
    description="Devuelve una lista paginada de todas las inferencias realizadas.",
)
async def history(
    page: int = Query(1, ge=1, description="Número de página"),
    limit: int = Query(20, ge=1, le=100, description="Elementos por página"),
    use_case: HistoryUseCase = Depends(get_history_use_case),
):
    result = await use_case.execute(page, limit)
    return HistoryResponse(
        data=result.data,
        total=result.total,
        page=result.page,
        limit=result.limit,
    )


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Re-entrenar K-Means",
    description="[Admin] Recibe un conjunto de imágenes y re-entrena el modelo K-Means. Requiere al menos K imágenes.",
)
async def retrain(
    k: int = Form(..., ge=2, le=20, description="Número de clústeres"),
    files: list[UploadFile] = File(..., description="Imágenes para re-entrenamiento (mínimo K)"),
    use_case: RetrainUseCase = Depends(get_retrain_use_case),
):
    if len(files) < k:
        raise HTTPException(
            status_code=400,
            detail=f"Se requieren al menos {k} imágenes para K={k}",
        )

    images = []
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{f.filename} no es una imagen")
        contents = await f.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail=f"{f.filename} está vacío")
        images.append((contents, f.filename or "unknown"))

    result = await use_case.execute(images, k)
    return RetrainResponse(
        k=result.k,
        silhouette=result.silhouette,
        davies_bouldin=result.davies_bouldin,
        inertia=result.inertia,
        mapping=result.mapping,
        trained_at=result.trained_at,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar estado del servicio",
    description="Indica si los modelos están cargados correctamente.",
)
async def health(
    encoder: TorchEncoderService = Depends(get_encoder),
    clustering: SklearnClusteringService = Depends(get_clustering),
    mapper: JsonClusterMapper = Depends(get_mapper),
):
    return HealthResponse(
        status="ok",
        model_loaded=True,
        k_value=None,
    )
