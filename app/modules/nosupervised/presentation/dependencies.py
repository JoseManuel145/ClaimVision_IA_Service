from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import get_session
from app.modules.nosupervised.infra.db.repository import PostgresInferenceRepository
from app.modules.nosupervised.infra.ml.preprocessor import TorchImagePreprocessor
from app.modules.nosupervised.infra.ml.encoder_service import TorchEncoderService
from app.modules.nosupervised.infra.ml.clustering_service import SklearnClusteringService
from app.modules.nosupervised.infra.mapping.cluster_mapper import JsonClusterMapper
from app.modules.nosupervised.application.predict_use_case import PredictUseCase
from app.modules.nosupervised.application.history_use_case import HistoryUseCase
from app.modules.nosupervised.application.retrain_use_case import RetrainUseCase


def get_repository(session: AsyncSession = Depends(get_session)) -> PostgresInferenceRepository:
    return PostgresInferenceRepository(session)


def get_preprocessor() -> TorchImagePreprocessor:
    import json
    from pathlib import Path
    config_path = Path(settings.MODELS_DIR) / "encoder_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        img_size = cfg.get("img_size", 224)
    except (FileNotFoundError, json.JSONDecodeError):
        img_size = 224
    return TorchImagePreprocessor(img_size=img_size)


def get_encoder() -> TorchEncoderService:
    return TorchEncoderService(settings.MODELS_DIR)


def get_clustering() -> SklearnClusteringService:
    return SklearnClusteringService(settings.MODELS_DIR)


def get_mapper() -> JsonClusterMapper:
    return JsonClusterMapper(settings.MODELS_DIR)


def get_predict_use_case(
    preprocessor=Depends(get_preprocessor),
    encoder=Depends(get_encoder),
    clustering=Depends(get_clustering),
    mapper=Depends(get_mapper),
    repository=Depends(get_repository),
) -> PredictUseCase:
    return PredictUseCase(preprocessor, encoder, clustering, mapper, repository)


def get_history_use_case(
    repository=Depends(get_repository),
) -> HistoryUseCase:
    return HistoryUseCase(repository)


def get_retrain_use_case() -> RetrainUseCase:
    return RetrainUseCase(settings.MODELS_DIR)
