import logging
import re

from app.modules.ocr.domain.models import IneData, PolizaData
from app.modules.ocr.domain.ports import DocumentStructuredExtractor

logger = logging.getLogger(__name__)


class RegexDocumentExtractor(DocumentStructuredExtractor):
    """Extracción de campos estructurados por regex, sin dependencia de LLM.

    Pipeline: OCR → Normalización → Segmentación → Clasificación → Extracción
    """

    # ── Look‑alike table para CURP ──────────────────────────────────────
    _DIGIT_LOOKALIKES = {
        "O": "0", "I": "1", "S": "5", "Z": "2", "G": "6", "B": "8", "L": "1",
    }

    # La regex suelta permite lookalikes en las posiciones numéricas del CURP
    # Posiciones: [4 letras][6 dígitos][H/M][2 letras estado][3 consonantes][2 alfanum]
    _CURP_DIGIT_CLASS = r"[0-9OISZGBL]"
    _CURP_LOOSE = re.compile(
        r"[A-Z]{4}"
        + _CURP_DIGIT_CLASS * 6
        + r"[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}"
    )
    _CURP_STRICT = re.compile(r"^[A-Z]{4}\d{6}[HM][A-Z]{2}[A-Z]{3}[A-Z0-9]{2}$")

    _MRZ_FILLER = "<>KX"

    _INE_LABELS = [
        "CLAVE", "CURP", "RFC", "MUNICIPIO", "SECCION", "SECCIÓN",
        "AÑO", "REGISTRO", "VOTO", "EMISION", "EMISIÓN",
    ]

    _MONTH_ABBR = {
        "ene": "01", "feb": "02", "mar": "03", "abr": "04",
        "may": "05", "jun": "06", "jul": "07", "ago": "08",
        "sep": "09", "oct": "10", "nov": "11", "dic": "12",
    }

    # ═══════════════════════════════════════════════════════════════════
    #  NORMALIZACIÓN Y SEGMENTACIÓN (pipeline base)
    # ═══════════════════════════════════════════════════════════════════

    def _normalize_text(self, text: str) -> str:
        """Normalización ligera: mayúsculas, colapsar espacios
        por línea, preservar saltos de línea."""
        text = text.upper()
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n +", "\n", text)
        return text.strip()

    def _to_blocks(self, text: str) -> list[str]:
        """Segmentación semántica: agrupa líneas en bloques basándose
        en etiquetas de sección. Elimina líneas muy cortas."""
        raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
        blocks: list[str] = []
        current: list[str] = []
        for line in raw_lines:
            if len(line) < 4:
                continue
            if re.match(r"^[A-Z\s]{3,}:?$", line):
                if current:
                    blocks.append(" ".join(current))
                    current = []
            current.append(line)
        if current:
            blocks.append(" ".join(current))
        return blocks

    # ═══════════════════════════════════════════════════════════════════
    #  VALIDACIONES SEMÁNTICAS
    # ═══════════════════════════════════════════════════════════════════

    def _validate_curp(self, curp: str) -> bool:
        if not self._CURP_STRICT.match(curp):
            return False
        try:
            mm = int(curp[6:8])
            dd = int(curp[8:10])
            if not (1 <= mm <= 12 and 1 <= dd <= 31):
                return False
        except (ValueError, IndexError):
            return False
        return True

    def _validate_rfc(self, rfc: str) -> bool:
        return bool(re.match(r"^[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{2,3}$", rfc))

    def _parse_date(self, raw: str) -> str:
        raw = raw.lower().strip()
        m = re.match(r"(\d{1,2})/([a-z]{3})/(\d{4})", raw)
        if m:
            dd, mon, yy = m.groups()
            mm = self._MONTH_ABBR.get(mon, "01")
            if mm:
                return f"{yy}-{mm}-{dd.zfill(2)}"
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
        if m:
            dd, mm, yy = m.groups()
            return f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"
        return ""

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS GENÉRICOS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _extract_by_label(
        cls,
        text: str,
        label_variants: list[str],
        *,
        max_len: int = 200,
    ) -> str:
        pattern = re.compile(
            r"(?i)(?:" + "|".join(label_variants) + r")\s*[:\-]?\s*(.+)",
        )
        for line in text.splitlines():
            m = pattern.search(line)
            if m:
                value = cls._clean(m.group(1))
                if value:
                    return value[:max_len]
        return ""

    @classmethod
    def _extract_date_by_label(
        cls,
        text: str,
        label_variants: list[str],
    ) -> str:
        pattern = re.compile(
            r"(?i)(?:" + "|".join(label_variants) + r")"
            + r"\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        )
        m = pattern.search(text)
        if not m:
            return ""
        raw = m.group(1).replace("-", "/")
        parts = raw.split("/")
        if len(parts) == 3:
            dd, mm, yy = parts
            if len(yy) == 2:
                yy = ("20" if int(yy) <= 50 else "19") + yy
            return f"{yy}-{mm.zfill(2)}-{dd.zfill(2)}"
        return ""

    # ═══════════════════════════════════════════════════════════════════
    #  INE
    # ═══════════════════════════════════════════════════════════════════

    async def extract_ine(self, text: str) -> IneData:
        text = self._normalize_text(text)
        logger.debug("INE texto normalizado (%d chars): %.200s…", len(text), text)

        # ── Parsear MRZ del reverso de la INE (TD1, 3 líneas de ~30 chars) ──
        mrz_data = self._parse_mrz_td1(text)
        if mrz_data:
            logger.debug("MRZ parseado: %s", mrz_data)

        curp = self._extract_curp_from_text(text)
        logger.debug("CURP extraído: %s", curp or "(vacío)")

        sexo = self._extract_sexo_from_text(text)
        if not sexo and curp:
            sexo = self._infer_sexo_from_curp(curp)
        # Fallback: sexo desde MRZ
        if not sexo and mrz_data.get("sexo"):
            sexo = mrz_data["sexo"]
        logger.debug("Sexo extraído: %s", sexo or "(vacío)")

        clave = self._extract_clave_elector_from_text(text)
        logger.debug("Clave elector extraída: %s", clave or "(vacío)")

        nombre = self._extract_nombre_robusto(text)
        logger.debug("Nombre extraído: %s", nombre or "(vacío)")

        credencial = self._extract_numero_credencial_from_text(text, curp)
        # Fallback: credencial desde MRZ (IDMEX line)
        if not credencial and mrz_data.get("credencial"):
            credencial = mrz_data["credencial"]
        logger.debug("Número credencial extraído: %s", credencial or "(vacío)")

        domicilio = self._extract_domicilio_from_text(text)
        logger.debug("Domicilio extraído: %s", domicilio or "(vacío)")

        rfc = self._extract_rfc_from_text(text, curp)
        logger.debug("RFC extraído: %s", rfc or "(vacío)")

        fecha = self._extract_fecha_nacimiento(text, curp)
        # Fallback: fecha desde MRZ
        if not fecha and mrz_data.get("fecha_nacimiento"):
            fecha = mrz_data["fecha_nacimiento"]
        logger.debug("Fecha nacimiento extraída: %s", fecha or "(vacío)")

        return IneData(
            nombre_completo=nombre,
            curp=curp,
            rfc=rfc or None,
            fecha_nacimiento=fecha,
            sexo=sexo,
            domicilio=domicilio,
            clave_elector=clave,
            numero_credencial=credencial or None,
        )

    # ── INE: MRZ TD1 (reverso de la INE) ──────────────────────────────

    def _parse_mrz_td1(self, text: str) -> dict:
        """Parsea las líneas MRZ del reverso de la INE mexicana (formato TD1).

        Formato TD1 de la INE:
        L1: IDMEX[no_credencial]<<[check][CIC_extra_digits]
        L2: [YYMMDD_nac][check][sexo][YYMMDD_exp][check][nacionalidad]...
        L3: [APELLIDO]<[APELLIDO]<<[NOMBRE(S)]<<<...

        El nombre ya se extrae en _extract_nombre_from_mrz, aquí nos enfocamos
        en las líneas L1 y L2 para extraer datos adicionales.
        """
        result = {}
        lines = text.upper().splitlines()

        mrz_lines = []
        for line in lines:
            # Limpiar la línea: quitar espacios y caracteres no-MRZ
            clean = re.sub(r"[^A-Z0-9<>/]", "", line)
            # Las líneas MRZ de TD1 tienen ~30 chars y contienen << o IDMEX
            if len(clean) >= 20 and ("<<" in clean or "IDMEX" in clean):
                mrz_lines.append(clean)

        # ── Línea 1: IDMEX + número de credencial ──
        for mline in mrz_lines:
            m = re.search(r"IDMEX(\d{10,13})", mline)
            if m:
                result["credencial"] = m.group(1)
                logger.debug("MRZ L1 credencial: %s", result["credencial"])
                break

        # ── Línea 2: fecha nacimiento + sexo ──
        # Formato: YYMMDD[check]S[YYMMDD_exp][check][NAC]...
        # Buscamos la línea que empieza con 6 dígitos seguidos de un check digit
        # y luego H/M/F (sexo)
        for mline in mrz_lines:
            if "IDMEX" in mline:
                continue
            # Buscar patrón: dígitos de fecha + sexo
            m = re.search(r"(\d{6})\d([HMF<])(\d{6})", mline)
            if m:
                fecha_raw = m.group(1)  # YYMMDD
                sexo_char = m.group(2)
                try:
                    yy = int(fecha_raw[0:2])
                    mm = int(fecha_raw[2:4])
                    dd = int(fecha_raw[4:6])
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        century = "19" if yy > 50 else "20"
                        result["fecha_nacimiento"] = f"{century}{yy:02d}-{mm:02d}-{dd:02d}"
                        logger.debug("MRZ L2 fecha nac: %s", result["fecha_nacimiento"])
                except (ValueError, IndexError):
                    pass

                if sexo_char in ("H", "M", "F"):
                    result["sexo"] = "M" if sexo_char == "F" else sexo_char
                    logger.debug("MRZ L2 sexo: %s", result["sexo"])
                break

            # Patrón alternativo: buscar YYMMDD seguido de 'H' en la línea
            # (típico de INE donde el sexo se codifica con H/M)
            m2 = re.match(r"(\d{6})\d?([HM])", mline)
            if m2:
                fecha_raw = m2.group(1)
                sexo_char = m2.group(2)
                try:
                    yy = int(fecha_raw[0:2])
                    mm = int(fecha_raw[2:4])
                    dd = int(fecha_raw[4:6])
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        century = "19" if yy > 50 else "20"
                        result["fecha_nacimiento"] = f"{century}{yy:02d}-{mm:02d}-{dd:02d}"
                except (ValueError, IndexError):
                    pass
                result["sexo"] = sexo_char
                break

        return result

    # ── INE: CURP ──────────────────────────────────────────────────────

    def _extract_curp_from_text(self, text: str) -> str:
        # Estrategia 1: buscar con label CURP
        label_match = re.search(
            r"CURP\s*[:\-]?\s*([A-Z0-9]{14,20})", text.upper()
        )
        if label_match:
            candidate = self._clean_curp(label_match.group(1)[:18])
            if self._validate_curp(candidate):
                return candidate

        # Estrategia 2: patrón suelto en texto completo
        candidate = self._find_and_clean_curp(text.upper())
        if candidate:
            return candidate

        # Estrategia 3: colapsar espacios (OCR puede meter espacios dentro del CURP)
        collapsed = re.sub(r"\s+", "", text.upper())
        return self._find_and_clean_curp(collapsed)

    def _find_and_clean_curp(self, upper_text: str) -> str:
        match = self._CURP_LOOSE.search(upper_text)
        if not match:
            return ""
        candidate = self._clean_curp(match.group(0))
        if not self._validate_curp(candidate):
            return ""
        return candidate

    def _clean_curp(self, curp: str) -> str:
        if not curp:
            return ""
        curp = curp.upper().strip()
        if len(curp) >= 18:
            fixed = list(curp[:18])
            # Posiciones 4-9: deben ser dígitos (fecha YYMMDD)
            for i in range(4, 10):
                fixed[i] = self._fix_digit(fixed[i])
            # Posiciones 16-17: dígito verificador + homoclave
            fixed[16] = self._fix_digit(fixed[16])
            fixed[17] = self._fix_digit(fixed[17])
            curp = "".join(fixed)
        return curp

    def _fix_digit(self, ch: str) -> str:
        return self._DIGIT_LOOKALIKES.get(ch.upper(), ch)

    # ── INE: Sexo ──────────────────────────────────────────────────────

    def _extract_sexo_from_text(self, text: str) -> str:
        # Patrón principal: SEXO seguido de H o M
        match = re.search(
            r"SEXO\s*[:\-]?\s*[^A-Z0-9]{0,5}(H|M)\b", text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # Fallback: buscar F/M al final de una línea que contenga "SEXO"
        for line in text.splitlines():
            if re.search(r"SEXO", line, re.IGNORECASE):
                m = re.search(r"\b([HMF])\s*$", line)
                if m:
                    val = m.group(1).upper()
                    return "M" if val == "F" else val

        return ""

    def _infer_sexo_from_curp(self, curp: str) -> str:
        if len(curp) >= 11:
            c = curp[10].upper()
            if c in ("H", "M", "X"):
                return c
        return ""

    # ── INE: Clave de elector ──────────────────────────────────────────

    def _extract_clave_elector_from_text(self, text: str) -> str:
        upper = text.upper()

        # Estrategia 1: label explícito "CLAVE DE ELECTOR" o variantes OCR
        clave_patterns = [
            # Formato estándar
            r"CLAVE\s*(?:DE\s*)?ELECT[O0]R\s*[:\-]?\s*((?:[A-Z0-9]\s*){16,20})",
            # OCR puede juntar/separar diferente
            r"CLAVE\s*(?:DE\s*)?ELECT[O0]R\s*[:\-]?\s*(\S{18})",
            # "CLAVE" sola seguida de un valor de 18 chars
            r"CLAVE\s*[:\-]\s*((?:[A-Z0-9]\s*){16,20})",
            # Variantes OCR comunes: ELECIOR, ELECTOR con ruido
            r"CLAVE\s*(?:DE\s*)?E\s*L\s*E\s*C\s*[TI]\s*[O0]\s*R\s*[:\-]?\s*((?:[A-Z0-9]\s*){16,20})",
        ]

        for pattern in clave_patterns:
            match = re.search(pattern, upper)
            if match:
                candidate = re.sub(r"\s+", "", match.group(1))[:18]
                if len(candidate) == 18:
                    return self._fix_clave_elector(candidate)

        # Estrategia 2: buscar patrón de 18 chars alfanuméricos que luzca
        # como clave de elector (6 letras + 8 dígitos + 1 letra/dígito + 3 dígitos)
        clave_struct = re.compile(
            r"\b([A-Z]{6}\d{8}[A-Z0-9]\d{3})\b"
        )
        for line in upper.splitlines():
            clean_line = re.sub(r"\s+", "", line)
            m = clave_struct.search(clean_line)
            if m:
                candidate = m.group(1)
                # No confundir con CURP
                if candidate != self._find_and_clean_curp(candidate):
                    return self._fix_clave_elector(candidate)

        # Estrategia 3: buscar cerca de la label "CLAVE"
        lines = upper.splitlines()
        for i, line in enumerate(lines):
            if re.search(r"\bCLAVE\b", line):
                # Revisar la misma línea y las siguientes
                search_text = " ".join(lines[i:min(i + 3, len(lines))])
                search_text_clean = re.sub(r"\s+", "", search_text)
                # Buscar cualquier secuencia alfanumérica de 18 chars
                candidates = re.findall(r"[A-Z0-9]{18}", search_text_clean)
                for cand in candidates:
                    # Validar que tiene mezcla de letras y dígitos
                    alpha_count = sum(1 for c in cand if c.isalpha())
                    digit_count = sum(1 for c in cand if c.isdigit())
                    if 6 <= alpha_count <= 12 and 6 <= digit_count <= 12:
                        return self._fix_clave_elector(cand)

        return ""

    def _fix_clave_elector(self, clave: str) -> str:
        """Corrección posicional determinística.
        Estructura: LLLLLL AAMMDD EE S CCC
        Pos 0-5: letras. Pos 6-13: dígitos (fecha + estado).
        Pos 14: sexo (H/M). Pos 15-17: numérico."""
        if len(clave) != 18:
            return clave
        clave = list(clave)

        # Posiciones 6-13 deben ser dígitos
        digit_fix = {"O": "0", "I": "1", "S": "5", "Z": "2", "G": "6", "B": "8", "L": "1"}
        for i in range(6, 14):
            if not clave[i].isdigit():
                clave[i] = digit_fix.get(clave[i], clave[i])

        # Posición 14: sexo (H/M)
        if clave[14] not in ("H", "M"):
            if clave[14] == "4":
                clave[14] = "H"
            elif clave[14] == "0":
                clave[14] = "H"

        # Posiciones 15-17: dígitos
        for i in range(15, 18):
            if not clave[i].isdigit():
                clave[i] = digit_fix.get(clave[i], clave[i])

        return "".join(clave)

    # ── INE: Número de credencial ──────────────────────────────────────

    def _extract_numero_credencial_from_text(self, text: str, curp: str = "") -> str:
        """Extrae el número de credencial (CIC) de 9-13 dígitos.
        Evita confundir con CURP, clave de elector u otros campos numéricos."""
        # Estrategia 1: con label explícito
        label_patterns = [
            r"(?:No\.?\s*DE\s*)?CREDENCIAL\s*[:\-]?\s*(\d{9,13})",
            r"(?:NUMERO|NÚM(?:ERO)?|NO\.?)\s*(?:DE\s*)?CREDENCIAL\s*[:\-]?\s*(\d{9,13})",
            r"CIC\s*[:\-]?\s*(\d{9,13})",
            r"IDMEX\s*(\d{9,13})",
        ]
        for pattern in label_patterns:
            match = re.search(pattern, text.upper())
            if match:
                return self._fix_numero_credencial(match.group(1))

        # Estrategia 2: buscar patrones numéricos de 9-13 dígitos que no sean
        # parte de otros campos conocidos
        # Extraer dígitos del CURP para excluirlos
        curp_digits = curp[4:10] if len(curp) >= 10 else ""

        # Buscar todos los números de 9-13 dígitos
        candidates = re.findall(r"(?<!\d)(\d{9,13})(?!\d)", text)
        for candidate in candidates:
            # Excluir si es parte del CURP
            if curp_digits and curp_digits in candidate:
                continue
            # Excluir números que parecen fechas largas (YYYYMMDD...)
            if re.match(r"^(19|20)\d{2}(0[1-9]|1[0-2])", candidate):
                continue
            # Excluir si está precedido por un label de otro campo
            idx = text.find(candidate)
            if idx > 0:
                context = text[max(0, idx - 40):idx].upper()
                if re.search(r"(?:CURP|RFC|SECCI[OÓ]N|A[ÑN]O|FECHA|MUNICIPIO)\s*[:\-]?\s*$", context):
                    continue
            return self._fix_numero_credencial(candidate)

        return ""

    def _fix_numero_credencial(self, num: str) -> str:
        """Corrección probabilística: genera candidatos por sustitución
        OCR y puntúa por consistencia local (repeticiones)."""
        if len(num) != 13 or not num.isdigit():
            return num
        confusion_map = {
            "5": ["3"], "3": ["5"],
            "8": ["6"], "6": ["8"],
        }
        candidates = [num]
        for i, c in enumerate(num):
            for alt in confusion_map.get(c, []):
                candidate = list(num)
                candidate[i] = alt
                candidates.append("".join(candidate))

        def score(n: str) -> int:
            return sum(1 for j in range(len(n) - 1) if n[j] == n[j + 1])

        return max(candidates, key=score)

    # ── INE: Nombre robusto ────────────────────────────────────────────

    def _extract_nombre_robusto(self, text: str) -> str:
        """MRZ → label NOMBRE → heurística multi-línea (2-5 palabras
        largas, sin dígitos, sin labels conocidos)."""
        name = self._extract_nombre_from_mrz(text)
        if name:
            return name

        # Buscar por label NOMBRE con posible valor en la misma línea
        name = self._extract_nombre_by_label(text)
        if name:
            return name

        # Heurística: línea con 2-5 palabras largas, solo letras
        _skip = re.compile(
            r"(?i)^(?:CLAVE|CURP|RFC|SEXO|DOMICILIO|AÑO|SECC|"
            r"MUNICIPIO|CREDENCIAL|REGISTRO|VOTO|EMISI|ESTADO|"
            r"LOCALIDAD|FECHA|VIGENCIA|INE|IFE|INSTITUTO|"
            r"ELECTORAL|MEXICO|IDMEX|NACIONAL)"
        )
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or _skip.search(stripped):
                continue
            words = stripped.split()
            if 2 <= len(words) <= 5 and all(len(w) > 2 for w in words):
                if not re.search(r"\d", stripped):
                    # Verificar que no sea una línea MRZ
                    if not re.search(r"[<>]{2,}", stripped):
                        return stripped.title()
        return ""

    def _extract_nombre_by_label(self, text: str) -> str:
        """Extrae nombre buscando labels comunes de la INE.
        Prioridad: APELLIDO PATERNO/MATERNO + NOMBRE(S) → NOMBRE simple."""
        lines = text.splitlines()
        _skip_value = re.compile(
            r"(?i)^(?:CLAVE|CURP|RFC|SEXO|DOMICILIO|AÑO|SECC|"
            r"MUNICIPIO|CREDENCIAL|REGISTRO|VOTO|EMISI|FECHA|"
            r"ESTADO|LOCALIDAD|VIGENCIA)"
        )

        # ── Prioridad 1: formato APELLIDO PATERNO / MATERNO / NOMBRE(S) ──
        apellido_p = ""
        apellido_m = ""
        nombre_s = ""

        for i, line in enumerate(lines):
            if re.search(r"(?i)APELLIDO\s*PATERNO", line):
                m = re.search(r"(?i)APELLIDO\s*PATERNO\s*[:\-]?\s*(.+)", line)
                if m:
                    apellido_p = self._clean(m.group(1))
                elif i + 1 < len(lines):
                    apellido_p = self._clean(lines[i + 1])

            if re.search(r"(?i)APELLIDO\s*MATERNO", line):
                m = re.search(r"(?i)APELLIDO\s*MATERNO\s*[:\-]?\s*(.+)", line)
                if m:
                    apellido_m = self._clean(m.group(1))
                elif i + 1 < len(lines):
                    apellido_m = self._clean(lines[i + 1])

            if re.search(r"(?i)NOMBRES?\s*(?:\(S\))?", line) and not re.search(r"(?i)APELLIDO", line):
                m = re.search(r"(?i)NOMBRES?\s*(?:\(S\))?\s*[:\-]?\s*(.+)", line)
                if m:
                    val = self._clean(m.group(1))
                    # Limpiar ruido OCR residual
                    val = re.sub(r"^\(?S?\)?\s*[:\-]?\s*", "", val).strip()
                    if val:
                        nombre_s = val
                elif i + 1 < len(lines):
                    nombre_s = self._clean(lines[i + 1])

        if nombre_s and (apellido_p or apellido_m):
            parts = [p for p in [nombre_s, apellido_p, apellido_m] if p]
            full = " ".join(parts)
            # Limpiar dígitos sueltos que podrían haberse colado
            full = re.sub(r"\d+", "", full).strip()
            if len(full) > 3:
                return full.title()

        # ── Prioridad 2: label NOMBRE simple (valor en la misma línea) ──
        for i, line in enumerate(lines):
            # "NOMBRE" o "NOMBRE(S)" seguido del valor en la misma línea
            m = re.search(r"(?i)NOMBRES?\s*(?:\(S\))?\s*[:\-]?\s*(.+)", line)
            if m:
                # Ignorar si la línea tiene APELLIDO (ya se procesó arriba)
                if re.search(r"(?i)APELLIDO", line):
                    continue
                value = self._clean(m.group(1))
                # Quitar posibles labels residuales y ruido OCR
                value = re.sub(r"(?i)\b(APELLIDO|PATERNO|MATERNO)\b\s*[:\-]?\s*", "", value).strip()
                value = re.sub(r"^\(?S?\)?\s*[:\-]?\s*", "", value).strip()
                if value and len(value) > 3:
                    return value.title()

            # "NOMBRE" solo, valor en la siguiente línea
            if re.search(r"(?i)^\s*NOMBRES?\s*(?:\(S\))?\s*[:\-]?\s*$", line):
                for j in range(i + 1, min(i + 3, len(lines))):
                    candidate = self._clean(lines[j])
                    if candidate and not _skip_value.match(candidate) and len(candidate) > 3:
                        if not re.search(r"\d", candidate):
                            return candidate.title()
                        break

        return ""

    def _extract_nombre_from_mrz(self, text: str) -> str:
        for line in text.upper().splitlines():
            compact = re.sub(r"[^A-Z<>KX]", "", line.strip())
            if not (26 <= len(compact) <= 36):
                continue
            if not re.fullmatch(rf"[A-Z{self._MRZ_FILLER}]+", compact):
                continue
            fillers = list(re.finditer(rf"[{self._MRZ_FILLER}]{{2,}}", compact))
            if len(fillers) < 2:
                continue
            candidates = []
            for m in fillers:
                before = compact[: m.start()]
                after = compact[m.end():]
                if re.search(r"[A-Z]{2,}$", before) and re.search(
                    r"^[A-Z]{2,}", after
                ):
                    candidates.append(m)
            if not candidates:
                continue
            boundary = max(candidates, key=lambda m: len(m.group(0)))
            apellidos_block = compact[: boundary.start()]
            nombres_block = compact[boundary.end():]
            apellidos = " ".join(
                w
                for w in re.split(rf"[{self._MRZ_FILLER}]+", apellidos_block)
                if len(w) >= 2
            )
            nombres = " ".join(
                w
                for w in re.split(rf"[{self._MRZ_FILLER}]+", nombres_block)
                if len(w) >= 2
            )
            if apellidos and nombres:
                return f"{nombres.title()} {apellidos.title()}"
        return ""

    # ── INE: Domicilio ─────────────────────────────────────────────────

    def _extract_domicilio_from_text(self, text: str) -> str:
        """1. Label DOMICILIO → bloque hasta stop-labels
        2. CP-centrado (excluyendo MRZ)
        3. Scoring semántico por bloques"""
        lines = text.splitlines()
        _stop = re.compile(
            r"(?i)(?:CLAVE|CURP|RFC|AÑO|FECHA|SECC|MUNICIPIO|CREDENCIAL|"
            r"ESTADO\s*DE|LOCALIDAD\s*DE|VIGENCIA|EMISI[OÓ]N)",
        )
        _mrz = re.compile(r"(?i)IDMEX|<<")
        dom_re = re.compile(r"(?i)DO[MB]?I?C?I?L?I?O|vomc[io]")
        for i, line in enumerate(lines):
            if dom_re.search(line):
                parts = []
                for j in range(i, min(len(lines), i + 6)):
                    if _mrz.search(lines[j]):
                        continue
                    cleaned = self._clean_ine_text(lines[j])
                    if not cleaned:
                        continue
                    if j > i and _stop.search(cleaned):
                        break
                    cleaned = dom_re.sub("", cleaned).strip()
                    if cleaned:
                        parts.append(cleaned)
                if parts:
                    return ", ".join(parts)
        cp_re = re.compile(r"\b\d{5}\b")
        for i, line in enumerate(lines):
            if _mrz.search(line):
                continue
            if cp_re.search(line):
                parts = []
                count = 0
                for j in range(i - 1, max(-1, i - 4), -1):
                    if _mrz.search(lines[j]):
                        continue
                    cleaned = self._clean_ine_text(lines[j])
                    if not cleaned:
                        continue
                    if _stop.search(cleaned):
                        break
                    parts.insert(0, cleaned)
                    count += 1
                    if count >= 2:
                        break
                current = self._clean_ine_text(line)
                if current:
                    parts.append(current)
                count = 0
                for j in range(i + 1, min(len(lines), i + 6)):
                    if _mrz.search(lines[j]):
                        continue
                    cleaned = self._clean_ine_text(lines[j])
                    if not cleaned:
                        continue
                    if _stop.search(cleaned):
                        break
                    parts.append(cleaned)
                    count += 1
                    if count >= 1:
                        break
                if parts:
                    return ", ".join(parts)
        return self._extract_domicilio_v2(text)

    def _extract_domicilio_v2(self, text: str) -> str:
        """Fallback: scoring semántico por bloques (CP + keywords)."""
        blocks = self._to_blocks(text)
        candidates = []
        _addr_kw = re.compile(
            r"(?i)\b(?:CALLE|AV|AVENIDA|COL|COLONIA|MZ|LT|"
            r"PRIVADA|CALLEJON|BLVD|BOULEVARD|FRACC|FRACCIONAMIENTO|"
            r"NÚM|NUM|MANZANA|LOTE|INTERIOR|INT|EXT)\b"
        )
        for block in blocks:
            score = 0
            if re.search(r"\b\d{5}\b", block):
                score += 3
            if _addr_kw.search(block):
                score += 2
            if re.search(r"\d", block):
                score += 1
            if score >= 3:
                candidates.append((score, block))
        if not candidates:
            return ""
        best = max(candidates, key=lambda x: x[0])[1]
        return self._clean(best)

    @staticmethod
    def _clean_ine_text(text: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9ÁÉÍÓÚÑ,.\s]", " ", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = []
        for t in cleaned.split():
            if len(t) == 1:
                continue
            if t.islower():
                continue
            if len(t) == 2 and t.isalpha() and t not in (
                "AV", "NO", "CP", "SM", "SA", "CV",
            ):
                continue
            tokens.append(t)
        return " ".join(tokens)

    # ── INE: RFC ───────────────────────────────────────────────────────

    def _extract_rfc_from_text(self, text: str, curp: str = "") -> str:
        """Extrae el RFC evitando confundir con el CURP."""
        match = re.search(
            r"(?i)RFC\s*[:\-]?\s*([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{0,3})",
            text.upper(),
        )
        if not match:
            return ""
        rfc = match.group(1).strip()
        if not self._validate_rfc(rfc):
            return ""
        # Si el RFC es idéntico a los primeros 13 chars del CURP, es válido (así funciona)
        # pero evitar devolver un CURP truncado como RFC
        if curp and len(rfc) >= 13 and rfc == curp[:len(rfc)]:
            # Es válido, RFC se deriva del CURP en México
            return rfc
        return rfc

    # ── INE: Fecha de nacimiento ───────────────────────────────────────

    def _extract_fecha_nacimiento(self, text: str, curp: str) -> str:
        """Intenta extraer la fecha de nacimiento del texto.
        1. Label explícito 'FECHA DE NACIMIENTO'
        2. Derivar del CURP"""
        # Estrategia 1: label explícito con fecha numérica
        fecha_patterns = [
            # dd/mm/yyyy o dd-mm-yyyy
            r"(?:FECHA\s*(?:DE\s*)?NACIMIENTO|F(?:\.\s*)?NAC(?:IMIENTO)?)\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            # dd/mmm/yyyy (con mes abreviado)
            r"(?:FECHA\s*(?:DE\s*)?NACIMIENTO|F(?:\.\s*)?NAC(?:IMIENTO)?)\s*[:\-]?\s*(\d{1,2}[/\-][A-Z]{3}[/\-]\d{2,4})",
        ]

        for pattern in fecha_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw_date = m.group(1)
                parsed = self._parse_date(raw_date)
                if parsed:
                    return parsed
                # Intentar parsear dd/mm/yy o dd-mm-yyyy directamente
                raw = raw_date.replace("-", "/")
                parts = raw.split("/")
                if len(parts) == 3:
                    dd, mm_or_mon, yy = parts
                    # Si el mes es texto
                    mm = self._MONTH_ABBR.get(mm_or_mon.lower(), "")
                    if mm:
                        if len(yy) == 2:
                            yy = ("20" if int(yy) <= 50 else "19") + yy
                        return f"{yy}-{mm}-{dd.zfill(2)}"
                    # Si es numérico
                    if mm_or_mon.isdigit():
                        if len(yy) == 2:
                            yy = ("20" if int(yy) <= 50 else "19") + yy
                        return f"{yy}-{mm_or_mon.zfill(2)}-{dd.zfill(2)}"

        # Estrategia 2: derivar del CURP
        return self._extract_fecha_from_curp(curp)

    def _extract_fecha_from_curp(self, curp: str) -> str:
        if len(curp) >= 10:
            yy = int(curp[4:6])
            mm = curp[6:8]
            dd = curp[8:10]
            century = "19" if yy > 50 else "20"
            return f"{century}{yy:02d}-{mm}-{dd}"
        return ""

    # ═══════════════════════════════════════════════════════════════════
    #  PÓLIZA
    # ═══════════════════════════════════════════════════════════════════

    async def extract_poliza(self, text: str) -> PolizaData:
        text = self._normalize_text(text)
        return PolizaData(
            numero_poliza=self._extract_poliza_numero(text),
            aseguradora=self._extract_poliza_aseguradora(text),
            nombre_asegurado=self._extract_poliza_nombre(text),
            vehiculo_marca=self._extract_poliza_marca(text),
            vehiculo_modelo=self._extract_poliza_modelo(text),
            vehiculo_anio=self._extract_year(text),
            vehiculo_placas=self._extract_poliza_placas(text),
            vehiculo_vin=self._extract_poliza_vin(text),
            vehiculo_color="",
            vigencia_inicio=self._extract_poliza_fecha(text, "DESDE"),
            vigencia_fin=self._extract_poliza_fecha(text, "HASTA"),
        )

    def _extract_poliza_numero(self, text: str) -> str:
        for line in text.splitlines():
            m = re.search(
                r"(?i)(?:(?:N[O°º\.]+\s*(?:DE\s+)?P[ÓO]LIZA|"
                r"No\.\s*de\s*cliente)\s*[:\-]?\s*(\S+))",
                line,
            )
            if m and len(m.group(1)) > 4:
                return m.group(1).strip()
            m2 = re.search(
                r"(?i)P[ÓO]LIZA\s*[:\-]?\s*(\d{8,15})",
                line,
            )
            if m2:
                return m2.group(1).strip()
        m_fallback = re.search(r"(?i)P[ÓO]LIZA[\s\S]{0,20}?(\d{8,15})", text)
        if m_fallback:
            return m_fallback.group(1)
        return ""

    def _extract_poliza_aseguradora(self, text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.search(r"(?i)Apoderado", line.strip()):
                if i + 1 < len(lines):
                    val = self._clean(lines[i + 1])
                    if val and len(val) > 3:
                        return val[:100]
        return ""

    def _extract_poliza_nombre(self, text: str) -> str:
        lines = text.splitlines()
        _skip = re.compile(
            r"(?i)^(?:datos|vehiculo|vehículo|domicilio|serie|modelo|"
            r"marca|placa|color|motor|uso|servicio|r\.f\.c|telefono|"
            r"agente|asegurador|nombre|observac)"
        )
        for i, line in enumerate(lines):
            m = re.search(r"(?i)Nombre\s*/?\s*Name?\s*[:\-]\s*(.+)", line)
            if m:
                val = self._clean(m.group(1))
                if val:
                    return val[:100]
            if re.search(r"(?i)^Nombre\s*:\s*$", line.strip()):
                for j in range(i + 1, min(i + 4, len(lines))):
                    candidate = self._clean(lines[j])
                    if candidate and not _skip.match(candidate):
                        return candidate[:100]
        return ""

    def _extract_poliza_marca(self, text: str) -> str:
        return self._extract_poliza_vehicle_field(text, r"MARCA(?:\s*/\s*MAKE)?")

    def _extract_poliza_modelo(self, text: str) -> str:
        return self._extract_poliza_vehicle_field(text, r"MODELO(?:\s*/\s*MODEL)?")

    def _extract_poliza_placas(self, text: str) -> str:
        label = self._extract_poliza_vehicle_field(text, r"Placas?(?:\s*/\s*Plates?)?")
        if label:
            return label
        m = re.search(r"\b[A-Z]{3}\d{3}[A-Z0-9]?\b", text.upper())
        return m.group(0) if m else ""

    def _extract_poliza_vin(self, text: str) -> str:
        label = self._extract_poliza_vehicle_field(text, r"Serie(?:\s*/\s*VIN)?")
        if label:
            return label
        return self._extract_vin_pattern(text)

    def _extract_vin_pattern(self, text: str) -> str:
        matches = re.findall(r"\b[A-HJ-NPR-Z0-9]{17}\b", text.upper())
        return matches[0] if matches else ""

    def _extract_poliza_vehicle_field(self, text: str, label_pattern: str) -> str:
        bilingual_re = re.compile(
            r"(?i)" + label_pattern + r"\s*[:\-]\s*(.+)",
        )
        for line in text.splitlines():
            m = bilingual_re.search(line)
            if m:
                val = self._clean(m.group(1))
                if val:
                    return val[:100]
        bilingual_empty_re = re.compile(
            r"(?i)" + label_pattern + r"\s*[:\-]\s*$",
        )
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if bilingual_empty_re.search(line):
                for j in range(i + 1, min(i + 3, len(lines))):
                    candidate = self._clean(lines[j])
                    if candidate and not re.match(
                        r"(?i)^(?:Marca|Modelo|Serie|Placas?|VIN|Uso|Servicio|Motor|"
                        r"A[ÑN]O|Year|Descrip|Número|No\.|Color)",
                        candidate,
                    ):
                        return candidate[:100]
        label_only_re = re.compile(
            r"(?i)" + label_pattern + r"\s*[:\-]?\s*$",
        )
        for i, line in enumerate(lines):
            if label_only_re.search(line):
                for j in range(i + 1, min(i + 10, len(lines))):
                    candidate = self._clean(lines[j])
                    if not candidate:
                        continue
                    if re.match(
                        r"(?i)^(?:Marca|Modelo|Serie|Placas?|VIN|Uso|Servicio|Motor|"
                        r"A[ÑN]O|Year|Descrip|Número|No\.|Color|Estado|Ciudad|"
                        r"R\.?\s*F\.?\s*C|Domicilio|Nombre|Telefono|Agente|"
                        r"Datos|Veh[ií]culo|Conductores|Coberturas|Observac|"
                        r"Apoderado|P[oó]liza|Vigencia|Emisi[oó]n|Moneda|"
                        r"Endoso|Forma|cliente|Prima|Gastos|I\.V\.A|Precio)",
                    ):
                        continue
                    return candidate[:100]
        return ""

    def _extract_poliza_fecha(self, text: str, keyword: str) -> str:
        date_re = re.compile(
            r"(?i)" + keyword + r"\s*[:\-]?\s*"
            r"(\d{1,2})\s*/\s*([a-z]{3,})\s*/\s*(\d{4})",
        )
        m = date_re.search(text)
        if m:
            dd, mon, yyyy = m.group(1), m.group(2).lower(), m.group(3)
            mm = self._MONTH_ABBR.get(mon, "")
            if mm:
                return f"{yyyy}-{mm}-{dd.zfill(2)}"
        num_re = re.compile(
            r"(?i)" + keyword + r"\s*[:\-]?\s*"
            r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})",
        )
        m = num_re.search(text)
        if m:
            dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
            return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
        return ""

    @staticmethod
    def _extract_year(text: str) -> int:
        patterns = [
            r"\bA[ÑN]O\s*/\s*YEAR\s*[:\-]?\s*(\d{4})",
            r"\bA[ÑN]O\s*[:\-]?\s*(\d{4})\b",
            r"\bYEAR\s*[:\-]?\s*(\d{4})\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                year = int(m.group(1))
                if 1900 <= year <= 2099:
                    return year
        m = re.search(r"(?i)MODELO\s*/?\s*MODEL?\s*[:\-]?\s*(\d{4})", text)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2099:
                return year
        return 0
