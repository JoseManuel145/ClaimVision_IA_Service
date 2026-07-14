import json
import httpx
from app.modules.ocr.domain.models import PolizaData, IneData


POLIZA_PROMPT = """
Dado el siguiente texto extraido de una poliza de seguro vehicular mexicana, extrae los campos solicitados. Responde SOLO con un JSON.

Texto:
\"\"\"
{text}
\"\"\"

Campos a extraer (responde con un JSON object con exactamente estas llaves):
- numero_poliza: string (numero de poliza)
- aseguradora: string (nombre de la compania aseguradora)
- nombre_asegurado: string (nombre completo del titular)
- vehiculo_marca: string (marca del vehiculo)
- vehiculo_modelo: string (modelo del vehiculo)
- vehiculo_anio: integer (anio del vehiculo)
- vehiculo_placas: string (numero de placas)
- vehiculo_vin: string o null (numero de serie VIN si aparece)
- vehiculo_color: string o null (color del vehiculo)
- vigencia_inicio: string (fecha de inicio de vigencia en formato YYYY-MM-DD)
- vigencia_fin: string (fecha de fin de vigencia en formato YYYY-MM-DD)

Si un campo no se encuentra en el texto, usa una cadena vacia "" para strings o null para opcionales.
Si el texto no parece una poliza de seguro, responde con un JSON con todos los campos vacios/null."""


INE_PROMPT = """Dado el siguiente texto extraido de una credencial del INE (Instituto Nacional Electoral) de Mexico, extrae los campos solicitados. Responde SOLO con un JSON.

Texto:
\"\"\"
{text}
\"\"\"

Campos a extraer (responde con un JSON object con exactamente estas llaves):
- nombre_completo: string (nombre completo como aparece en la credencial)
- curp: string (CURP de 18 caracteres)
- rfc: string o null (RFC si aparece)
- fecha_nacimiento: string (fecha de nacimiento en formato YYYY-MM-DD)
- sexo: string ("H" o "M")
- domicilio: string (domicilio completo como aparece)
- clave_elector: string (clave de elector de 18 caracteres)
- numero_credencial: string o null (numero de credencial si aparece)

Si un campo no se encuentra en el texto, usa una cadena vacia "" para strings o null para opcionales.
Si el texto no parece una credencial INE, responde con un JSON con todos los campos vacios/null.
"""


class OllamaDocumentExtractor:
    def __init__(
        self,
        base_url: str,
        model: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def _call_ollama(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "{}")

    async def extract_poliza(self, text: str) -> PolizaData:
        prompt = POLIZA_PROMPT.format(text=text)
        raw = await self._call_ollama(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return PolizaData(
                numero_poliza="", aseguradora="", nombre_asegurado="",
                vehiculo_marca="", vehiculo_modelo="", vehiculo_anio=0,
                vehiculo_placas="", vehiculo_vin=None, vehiculo_color=None,
                vigencia_inicio="", vigencia_fin="",
            )

        return PolizaData(
            numero_poliza=data.get("numero_poliza", ""),
            aseguradora=data.get("aseguradora", ""),
            nombre_asegurado=data.get("nombre_asegurado", ""),
            vehiculo_marca=data.get("vehiculo_marca", ""),
            vehiculo_modelo=data.get("vehiculo_modelo", ""),
            vehiculo_anio=int(data.get("vehiculo_anio", 0) or 0),
            vehiculo_placas=data.get("vehiculo_placas", ""),
            vehiculo_vin=data.get("vehiculo_vin"),
            vehiculo_color=data.get("vehiculo_color"),
            vigencia_inicio=data.get("vigencia_inicio", ""),
            vigencia_fin=data.get("vigencia_fin", ""),
        )

    async def extract_ine(self, text: str) -> IneData:
        prompt = INE_PROMPT.format(text=text)
        raw = await self._call_ollama(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return IneData(
                nombre_completo="", curp="", rfc=None,
                fecha_nacimiento="", sexo="", domicilio="",
                clave_elector="", numero_credencial=None,
            )

        return IneData(
            nombre_completo=data.get("nombre_completo", ""),
            curp=data.get("curp", ""),
            rfc=data.get("rfc"),
            fecha_nacimiento=data.get("fecha_nacimiento", ""),
            sexo=data.get("sexo", ""),
            domicilio=data.get("domicilio", ""),
            clave_elector=data.get("clave_elector", ""),
            numero_credencial=data.get("numero_credencial"),
        )
