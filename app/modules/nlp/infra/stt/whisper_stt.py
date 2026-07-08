import os
import subprocess
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel


class WhisperSTTService:
    _model = None

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    async def transcribir(self, audio_bytes: bytes, filename: str) -> tuple[str, float]:
        model = self._get_model()
        suffix = Path(filename).suffix if Path(filename).suffix else ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_wav_path = tmp_in_path.rsplit(".", 1)[0] + ".wav"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    tmp_wav_path,
                ],
                capture_output=True,
                check=True,
            )
            segments, info = model.transcribe(tmp_wav_path, language="es")
            texto = " ".join(segment.text for segment in segments)
            duracion = round(info.duration, 2) if info.duration else 0.0
            return texto, duracion
        finally:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)

    def _get_model(self) -> WhisperModel:
        if WhisperSTTService._model is None:
            WhisperSTTService._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return WhisperSTTService._model
