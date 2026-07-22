from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_session, async_session_factory
from app.modules.nlp.infra.db.repository import PostgresVozRepository
from app.modules.nlp.infra.stt.groq_stt import GroqSTTService
from app.core.config import settings
from app.modules.nlp.infra.llm.groq_extractor import GroqExtractor
from app.modules.nlp.application.transcribir_use_case import TranscribirUseCase
from app.modules.nlp.application.transcripcion_job_use_case import TranscripcionJobUseCase
from app.modules.nlp.application.history_use_case import HistoryUseCase


def get_nlp_repository(
    session: AsyncSession = Depends(get_session),
) -> PostgresVozRepository:
    return PostgresVozRepository(session)


def get_stt_service() -> GroqSTTService:
    return GroqSTTService(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_WHISPER_MODEL,
    )


def get_llm_service() -> GroqExtractor:
    return GroqExtractor(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_LLM_MODEL,
    )


def get_transcribir_use_case(
    repo: PostgresVozRepository = Depends(get_nlp_repository),
    stt: GroqSTTService = Depends(get_stt_service),
    llm: GroqExtractor = Depends(get_llm_service),
) -> TranscribirUseCase:
    return TranscribirUseCase(repo, stt, llm)


def get_transcripcion_job_use_case(
    stt: GroqSTTService = Depends(get_stt_service),
    llm: GroqExtractor = Depends(get_llm_service),
) -> TranscripcionJobUseCase:
    return TranscripcionJobUseCase(stt, llm, async_session_factory)


def get_history_use_case(
    repo: PostgresVozRepository = Depends(get_nlp_repository),
) -> HistoryUseCase:
    return HistoryUseCase(repo)
