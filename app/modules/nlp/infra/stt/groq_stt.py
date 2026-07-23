import tempfile
from pathlib import Path
import httpx
from groq import Groq


class GroqSTTService:
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        ssl_verify: bool = True,
    ):
        http_client = httpx.Client(verify=ssl_verify, timeout=120.0)
        self.client = Groq(api_key=api_key, http_client=http_client)
        self.model = model

    async def transcribir(self, audio_bytes: bytes, filename: str) -> tuple[str, float]:
        suffix = Path(filename).suffix if Path(filename).suffix else ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    file=(filename, audio_file),
                    model=self.model,
                    language="es",
                    response_format="verbose_json",
                )
            texto = result.text
            duracion = round(result.duration, 2) if hasattr(result, "duration") and result.duration else 0.0
            return texto, duracion
        finally:
            import os
            os.unlink(tmp_path)
