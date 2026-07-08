from uuid import uuid4
from datetime import datetime, timezone
from app.modules.nlp.domain.models import VozTranscripcion
from app.modules.nlp.domain.ports import SpeechToTextService, NlpAnalysisService, VozRepository


class TranscribirUseCase:
    def __init__(
        self,
        repo: VozRepository,
        stt: SpeechToTextService,
        llm: NlpAnalysisService,
    ):
        self._repo = repo
        self._stt = stt
        self._llm = llm

    async def execute(self, audio_bytes: bytes, filename: str) -> VozTranscripcion:
        texto, duracion = await self._stt.transcribir(audio_bytes, filename)
        entidades = await self._llm.extraer_danos(texto)
        transcripcion = VozTranscripcion(
            id=str(uuid4()),
            filename=filename,
            texto=texto,
            duracion_seg=duracion,
            entidades=entidades,
            created_at=datetime.now(timezone.utc),
        )
        saved = await self._repo.save(transcripcion)
        return saved
