from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session
from app.modules.nlp.infra.db.repository import PostgresVozRepository
from app.modules.nlp.infra.stt.whisper_stt import WhisperSTTService
from app.core.config import settings
from app.modules.nlp.infra.llm.ollama_extractor import OllamaExtractor
from app.modules.nlp.application.transcribir_use_case import TranscribirUseCase
from app.modules.nlp.application.history_use_case import HistoryUseCase


def get_nlp_repository(
    session: AsyncSession = Depends(get_session),
) -> PostgresVozRepository:
    return PostgresVozRepository(session)


def get_stt_service() -> WhisperSTTService:
    return WhisperSTTService()


def get_llm_service() -> OllamaExtractor:
    return OllamaExtractor(
        base_url=settings.OLLAMA_URL,
        model=settings.OLLAMA_MODEL,
    )


def get_transcribir_use_case(
    repo: PostgresVozRepository = Depends(get_nlp_repository),
    stt: WhisperSTTService = Depends(get_stt_service),
    llm: OllamaExtractor = Depends(get_llm_service),
) -> TranscribirUseCase:
    return TranscribirUseCase(repo, stt, llm)


def get_history_use_case(
    repo: PostgresVozRepository = Depends(get_nlp_repository),
) -> HistoryUseCase:
    return HistoryUseCase(repo)
