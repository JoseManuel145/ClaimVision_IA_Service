import json
import logging
import re
import httpx
from app.modules.ocr.domain.models import PolizaData, IneData
logger = logging.getLogger(__name__)


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
Si el texto no parece una poliza de seguro, responde con un JSON con todos los campos vacios/null.
"""


INE_PROMPT = """Eres un sistema experto en extracción de datos estructurados a partir de texto OCR de documentos de identidad mexicanos (INE). 
Tu ÚNICA tarea es extraer información específica y formatearla estrictamente como un objeto JSON válido. No incluyas saludos, explicaciones, ni texto en formato Markdown fuera del JSON.

REGLAS DE EXTRACCIÓN:
1. "nombre_completo": Busca la etiqueta "NOMBRE". Une el apellido paterno, apellido materno y nombre(s) en una sola cadena, separados por un espacio.
2. "domicilio": Busca la etiqueta "DOMICILIO" y extrae todo el texto hasta encontrar otra etiqueta importante (como CLAVE DE ELECTOR, CURP, o MUNICIPIO). Incluye calle, número, colonia, C.P., ciudad y estado.
3. "rfc": Es un código alfanumérico de 10 a 13 caracteres. A veces el OCR lo confunde con la CURP (18 caracteres). Si encuentras algo que coincida con el formato de RFC, extráelo; de lo contrario, devuelve null.
4. "numero_credencial": Suele estar en la parte inferior o trasera (código de 12 a 13 dígitos) o referirse al CIC. Si no estás seguro, devuelve null.

EJEMPLO DE ENTRADA:
Texto:
INSTITUTO NACIONAL ELECTORAL
CREDENCIAL PARA VOTAR
NOMBRE
GARCIA
LOPEZ
MARIA FERNANDA
DOMICILIO
C FLORES MAGON 123 COL CENTRO
CP 06000 CUAUHTEMOC CIUDAD DE MEXICO
CLAVE DE ELECTOR GRLPMR90010109H100
CURP GALM900101MDFRRN09
RFC GALM900101XYZ
AÑO DE REGISTRO 2010

EJEMPLO DE SALIDA ESPERADA:
{{
  "nombre_completo": "GARCIA LOPEZ MARIA FERNANDA",
  "domicilio": "C FLORES MAGON 123 COL CENTRO CP 06000 CUAUHTEMOC CIUDAD DE MEXICO",
  "rfc": "GALM900101XYZ",
  "numero_credencial": null
}}

DATOS A PROCESAR:
Texto:
\"\"\"
{text}
\"\"\"

Responde ÚNICAMENTE con el JSON resultante usando la siguiente estructura, sin bloques de código (```json) ni texto adicional:
{{
  "nombre_completo": "",
  "domicilio": "",
  "rfc": null,
  "numero_credencial": null
}}"""


class OllamaDocumentExtractor:
    # Caracteres que el OCR confunde tipicamente con un digito. Se usan
    # tanto para "reparar" un CURP ya encontrado como para ampliar la
    # busqueda inicial (el regex estricto original fallaba completo en
    # cuanto un digito de la franja de fecha venia OCR'eado como letra).
    _DIGIT_LOOKALIKES = {"O": "0", "I": "1", "S": "5", "Z": "2", "G": "6", "B": "8", "L": "1"}

    _CURP_LOOSE = re.compile(
        r"[A-Z]{4}[0-9" + "".join(_DIGIT_LOOKALIKES) + r"]{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}"
    )
    _CURP_STRICT = re.compile(r"^[A-Z]{4}\d{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}$")

    # Caracteres que el OCR suele escupir en vez del relleno "<" de la
    # zona MRZ (reverso del INE nuevo).
    _MRZ_FILLER = "<>KX"

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
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "{}")

    def _safe_json_parse(self, raw: str) -> dict:
        """format='json' deberia forzar JSON puro, pero algunos modelos
        igual envuelven la respuesta en ```json. Se limpia antes de
        tirar la respuesta a la basura, y se deja rastro en logs si de
        plano no se pudo parsear (antes se perdia en silencio)."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Respuesta del modelo no es JSON valido: %.200s", cleaned)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    async def extract_poliza(self, text: str) -> PolizaData:
        prompt = POLIZA_PROMPT.format(text=text)
        raw = await self._call_ollama(prompt)
        data = self._safe_json_parse(raw)

        if not data:
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
        nombre_from_mrz = self._extract_nombre_from_mrz(text)
        credencial_from_ocr = self._extract_numero_credencial_from_text(text)

        prompt = INE_PROMPT.format(text=text)
        raw = await self._call_ollama(prompt)
        data = self._safe_json_parse(raw)

        curp = curp_from_ocr or self._clean_curp(data.get("curp", ""))

        return IneData(
            nombre_completo=nombre_from_mrz or data.get("nombre_completo", ""),
            curp=curp,
            rfc=data.get("rfc"),
            fecha_nacimiento=self._extract_fecha_from_curp(curp),
            sexo=sexo_from_ocr,
            domicilio=data.get("domicilio", ""),
            clave_elector=clave_elector_from_ocr or data.get("clave_elector", ""),
            numero_credencial=credencial_from_ocr or data.get("numero_credencial"),
        )

    def _extract_curp_from_text(self, text: str) -> str:
        candidate = self._find_and_clean_curp(text.upper())
        if candidate:
            return candidate
        # El CURP a veces queda partido por un salto de linea que mete
        # el OCR entre bloques de la credencial. Segundo intento sin
        # espacios ni saltos de linea.
        collapsed = re.sub(r"\s+", "", text.upper())
        return self._find_and_clean_curp(collapsed)

    def _find_and_clean_curp(self, upper_text: str) -> str:
        match = self._CURP_LOOSE.search(upper_text)
        if not match:
            return ""
        candidate = self._clean_curp(match.group(0))
        return candidate if self._CURP_STRICT.match(candidate) else ""

    def _extract_sexo_from_text(self, text: str) -> str:
        match = re.search(r"SEXO\s*[:\-]?\s*[^A-Z0-9]{0,3}(H|M)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return ""

    def _extract_clave_elector_from_text(self, text: str) -> str:
        match = re.search(
            r"CLAVE\s*(?:DE\s*)?ELECTOR\s*[:\-]?\s*((?:[A-Z0-9]\s*){17,18})",
            text.upper(),
        )
        if not match:
            return ""
        candidate = re.sub(r"\s+", "", match.group(1))
        return candidate if len(candidate) == 18 else ""

    def _extract_numero_credencial_from_text(self, text: str) -> str:
        # Corrida aislada de 12-13 digitos: el numero OCR/credencial.
        # Respaldo determinista al dato que igual le pedimos al modelo.
        match = re.search(r"(?<!\d)\d{12,13}(?!\d)", text)
        return match.group(0) if match else ""

    def _extract_nombre_from_mrz(self, text: str) -> str:
        for line in text.upper().splitlines():
            compact = re.sub(r"[^A-Z<>KX]", "", line.strip())
            if not (26 <= len(compact) <= 31):
                continue
            if not re.fullmatch(rf"[A-Z{self._MRZ_FILLER}]+", compact):
                continue

            fillers = list(re.finditer(rf"[{self._MRZ_FILLER}]+", compact))
            if len(fillers) < 2:
                continue

            # El separador "<<" entre apellidos y nombres esta rodeado
            # de texto alfabetico a ambos lados. El padding final solo
            # tiene texto a la izquierda.
            candidates = []
            for m in fillers:
                before = compact[:m.start()]
                after = compact[m.end():]
                if re.search(r"[A-Z]{2,}$", before) and re.search(r"^[A-Z]{2,}", after):
                    candidates.append(m)
            if not candidates:
                continue
            boundary = max(candidates, key=lambda m: len(m.group(0)))
            apellidos_block = compact[: boundary.start()]
            nombres_block = compact[boundary.end():]

            apellidos = " ".join(
                w for w in re.split(rf"[{self._MRZ_FILLER}]+", apellidos_block) if len(w) >= 2
            )
            nombres = " ".join(
                w for w in re.split(rf"[{self._MRZ_FILLER}]+", nombres_block) if len(w) >= 2
            )
            if apellidos and nombres:
                return f"{nombres.title()} {apellidos.title()}"
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
        return self._DIGIT_LOOKALIKES.get(ch.upper(), ch)