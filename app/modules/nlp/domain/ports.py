from typing import Protocol
from app.modules.nlp.domain.models import VozTranscripcion, DamageEntity


class SpeechToTextService(Protocol):
    async def transcribir(self, audio_bytes: bytes, filename: str) -> tuple[str, float]: ...


class NlpAnalysisService(Protocol):
    async def extraer_danos(self, texto: str) -> list[DamageEntity]: ...


class VozRepository(Protocol):
    async def save(self, transcripcion: VozTranscripcion) -> VozTranscripcion: ...

    async def list_paginated(
        self, page: int, limit: int
    ) -> tuple[list[VozTranscripcion], int]: ...

    async def get_by_id(self, id: str) -> VozTranscripcion | None: ...
