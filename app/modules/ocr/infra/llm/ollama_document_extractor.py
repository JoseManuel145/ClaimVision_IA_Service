import json
import re
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


INE_PROMPT = """Extrae el nombre completo, domicilio, RFC y numero de credencial del texto de una credencial INE. Responde SOLO con un JSON.

Texto:
\"\"\"
{text}
\"\"\"

El nombre completo aparece despues de la etiqueta "NOMBRE" y puede estar en varias lineas con apellido paterno, apellido materno y nombre(s).
El domicilio aparece despues de "DOMICILIO" con calle, colonia, CP, ciudad y estado.
El RFC es un codigo alfanumerico de 10-13 caracteres cerca de la CURP.
El numero de credencial puede aparecer en la parte inferior de la credencial.

Campos:
- nombre_completo: string (apellido paterno, apellido materno, nombre(s))
- domicilio: string (domicilio completo con calle, colonia, CP, ciudad, estado)
- rfc: string o null (RFC si aparece)
- numero_credencial: string o null (numero de credencial si aparece)

Si un campo no se encuentra, usa "" para strings o null para opcionales."""


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
        curp_from_ocr = self._extract_curp_from_text(text)
        sexo_from_ocr = self._extract_sexo_from_text(text)
        if not sexo_from_ocr and curp_from_ocr:
            sexo_from_ocr = self._infer_sexo_from_curp(curp_from_ocr)
        clave_elector_from_ocr = self._extract_clave_elector_from_text(text)

        prompt = INE_PROMPT.format(text=text)
        raw = await self._call_ollama(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        curp = curp_from_ocr or self._clean_curp(data.get("curp", ""))

        return IneData(
            nombre_completo=data.get("nombre_completo", ""),
            curp=curp,
            rfc=data.get("rfc"),
            fecha_nacimiento=self._extract_fecha_from_curp(curp),
            sexo=sexo_from_ocr,
            domicilio=data.get("domicilio", ""),
            clave_elector=clave_elector_from_ocr or data.get("clave_elector", ""),
            numero_credencial=data.get("numero_credencial"),
        )

    def _extract_curp_from_text(self, text: str) -> str:
        match = re.search(
            r"[A-Z]{4}\d{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}",
            text.upper(),
        )
        if match:
            return self._clean_curp(match.group(0))
        return ""

    def _extract_sexo_from_text(self, text: str) -> str:
        match = re.search(r"SEXO\s*[:\-]?\s*(H|M)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    def _extract_clave_elector_from_text(self, text: str) -> str:
        match = re.search(r"CLAVE\s*(?:DE\s*)?ELECTOR\s*[:\-]?\s*([A-Z0-9]{18})", text.upper())
        if match:
            return match.group(1)
        return ""

    def _extract_fecha_from_curp(self, curp: str) -> str:
        if len(curp) >= 10:
            yy = int(curp[4:6])
            mm = curp[6:8]
            dd = curp[8:10]
            century = "19" if yy > 50 else "20"
            return f"{century}{yy:02d}-{mm}-{dd}"
        return ""

    def _infer_sexo_from_curp(self, curp: str) -> str:
        if len(curp) >= 11:
            c = curp[10].upper()
            if c in ("H", "M", "X"):
                return c
        return ""

    def _clean_curp(self, curp: str) -> str:
        if not curp:
            return ""
        curp = curp.upper().strip()
        if len(curp) >= 18:
            fixed = list(curp)
            for i in range(4, 10):
                fixed[i] = self._fix_digit(fixed[i])
            fixed[16] = self._fix_digit(fixed[16])
            fixed[17] = self._fix_digit(fixed[17])
            curp = "".join(fixed)
        return curp

    def _fix_digit(self, ch: str) -> str:
        mapping = {"O": "0", "I": "1", "S": "5", "Z": "2", "G": "6", "B": "8", "L": "1"}
        return mapping.get(ch.upper(), ch)
