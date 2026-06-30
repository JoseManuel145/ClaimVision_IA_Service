from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import get_session
from app.modules.supervised.infra.ml.preprocessor import SupervisedPreprocessor
from app.modules.supervised.infra.ml.classifier_service import ResNetClassifierService
from app.modules.supervised.infra.db.repository import PostgresV2PredictionRepository, PostgresRetrainJobRepository
from app.modules.supervised.application.predict_use_case import V2PredictUseCase
from app.modules.supervised.application.retrain_use_case import V2RetrainUseCase
from app.modules.supervised.application.history_use_case import V2HistoryUseCase


def get_v2_preprocessor() -> SupervisedPreprocessor:
    return SupervisedPreprocessor()


def get_classifier() -> ResNetClassifierService:
    return ResNetClassifierService(settings.MODELS_DIR)


def get_v2_prediction_repository(session: AsyncSession = Depends(get_session)) -> PostgresV2PredictionRepository:
    return PostgresV2PredictionRepository(session)


def get_v2_retrain_job_repository(session: AsyncSession = Depends(get_session)) -> PostgresRetrainJobRepository:
    return PostgresRetrainJobRepository(session)


def get_v2_predict_use_case(
    preprocessor=Depends(get_v2_preprocessor),
    classifier=Depends(get_classifier),
    repository=Depends(get_v2_prediction_repository),
) -> V2PredictUseCase:
    return V2PredictUseCase(preprocessor, classifier, repository)


def get_v2_retrain_use_case(
    classifier=Depends(get_classifier),
    job_repo=Depends(get_v2_retrain_job_repository),
) -> V2RetrainUseCase:
    return V2RetrainUseCase(classifier, job_repo)


def get_v2_history_use_case(
    repository=Depends(get_v2_prediction_repository),
) -> V2HistoryUseCase:
    return V2HistoryUseCase(repository)
