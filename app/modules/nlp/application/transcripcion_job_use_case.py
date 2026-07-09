import asyncio
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.modules.nlp.domain.models import TranscripcionJob, VozTranscripcion
from app.modules.nlp.domain.ports import SpeechToTextService, NlpAnalysisService
from app.modules.nlp.infra.db.repository import (
    PostgresTranscripcionJobRepository,
    PostgresVozRepository,
)


class TranscripcionJobUseCase:
    def __init__(
        self,
        stt: SpeechToTextService,
        llm: NlpAnalysisService,
        session_factory: async_sessionmaker[AsyncSession],
    ):
        self._stt = stt
        self._llm = llm
        self._session_factory = session_factory

    async def start_job(self, audio_bytes: bytes, filename: str) -> TranscripcionJob:
        now = datetime.now(timezone.utc)
        job = TranscripcionJob(
            id=str(uuid4()),
            filename=filename,
            status="pending",
            progress=0,
            result_id=None,
            error=None,
            created_at=now,
            updated_at=now,
        )
        async with self._session_factory() as session:
            repo = PostgresTranscripcionJobRepository(session)
            await repo.save(job)

        asyncio.create_task(self._process(job.id, audio_bytes, filename))
        return job

    async def get_job_status(self, job_id: str) -> Optional[TranscripcionJob]:
        async with self._session_factory() as session:
            repo = PostgresTranscripcionJobRepository(session)
            return await repo.get_by_id(job_id)

    async def _process(self, job_id: str, audio_bytes: bytes, filename: str):
        async with self._session_factory() as session:
            job_repo = PostgresTranscripcionJobRepository(session)
            voz_repo = PostgresVozRepository(session)

            try:
                await self._set_job(job_repo, job_id, status="processing", progress=10)

                texto, duracion = await self._stt.transcribir(audio_bytes, filename)
                await self._set_job(job_repo, job_id, progress=60)

                entidades = await self._llm.extraer_danos(texto)
                await self._set_job(job_repo, job_id, progress=90)

                transcripcion = VozTranscripcion(
                    id=str(uuid4()),
                    filename=filename,
                    texto=texto,
                    duracion_seg=duracion,
                    entidades=entidades,
                    created_at=datetime.now(timezone.utc),
                )
                saved = await voz_repo.save(transcripcion)
                await self._set_job(
                    job_repo, job_id,
                    progress=100, status="completed", result_id=saved.id,
                )

            except Exception as e:
                await self._set_job(job_repo, job_id, status="failed", error=str(e))

    async def _set_job(
        self,
        repo: PostgresTranscripcionJobRepository,
        job_id: str,
        **kwargs,
    ):
        job = await repo.get_by_id(job_id)
        if not job:
            return
        for key, value in kwargs.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
        await repo.update(job)
