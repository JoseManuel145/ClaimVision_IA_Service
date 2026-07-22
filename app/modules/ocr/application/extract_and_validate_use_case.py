from app.modules.ocr.domain.models import PolizaData, IneData
from app.modules.ocr.application.extract_poliza_use_case import ExtractPolizaUseCase
from app.modules.ocr.application.extract_ine_use_case import ExtractIneUseCase


class ExtractAndValidateUseCase:
    def __init__(
        self,
        extract_poliza: ExtractPolizaUseCase,
        extract_ine: ExtractIneUseCase,
    ):
        self._extract_poliza = extract_poliza
        self._extract_ine = extract_ine

    async def execute(
        self,
        poliza_bytes: bytes,
        poliza_filename: str,
        ine_bytes: bytes,
        ine_filename: str,
        ine_content_type: str,
    ) -> dict:
        poliza_extraction = await self._extract_poliza.execute(
            poliza_bytes, poliza_filename
        )
        ine_extraction = await self._extract_ine.execute(
            ine_bytes, ine_filename, ine_content_type
        )

        poliza_data = PolizaData(**poliza_extraction.extracted_data)
        ine_data = IneData(**ine_extraction.extracted_data)

        validation = self._cross_validate(poliza_data, ine_data)

        return {
            "poliza": poliza_extraction,
            "ine": ine_extraction,
            "validation": validation,
        }

    def _cross_validate(self, poliza: PolizaData, ine: IneData) -> dict:
        import re
        import unicodedata

        detalles = []

        # ── Normalización de Nombres ──
        def clean_name(name: str) -> str:
            if not name:
                return ""
            # Convertir a minúsculas y remover acentos
            name = name.lower()
            name = "".join(
                c for c in unicodedata.normalize("NFD", name)
                if unicodedata.category(c) != "Mn"
            )
            return re.sub(r"[^\w\s]", "", name).strip()

        poliza_clean = clean_name(poliza.nombre_asegurado)
        ine_clean = clean_name(ine.nombre_completo)

        # ── Comparación de tokens ──
        stop_words = {"de", "del", "la", "los", "y"}
        tokens_poliza = {w for w in poliza_clean.split() if w not in stop_words}
        tokens_ine = {w for w in ine_clean.split() if w not in stop_words}

        nombre_match = False
        if tokens_poliza and tokens_ine:
            intersection = tokens_poliza.intersection(tokens_ine)
            min_tokens = min(len(tokens_poliza), len(tokens_ine))
            
            # Coinciden si se comparten todos los tokens del nombre más corto,
            # o si hay al menos 2 tokens en común (ej. Nombre + Apellido).
            if len(intersection) >= min_tokens or len(intersection) >= 2:
                nombre_match = True
            else:
                detalles.append(
                    f"Nombre no coincide: Póliza='{poliza.nombre_asegurado}' vs INE='{ine.nombre_completo}'"
                )
        else:
            detalles.append("No se pudieron extraer nombres válidos para comparar.")

        # ── Consistencia entre CURP y RFC ──
        curp_rfc_match = True
        if ine.curp and ine.rfc:
            curp_clean = re.sub(r"[^A-Z0-9]", "", ine.curp.upper())
            rfc_clean = re.sub(r"[^A-Z0-9]", "", ine.rfc.upper())
            
            if len(curp_clean) >= 10 and len(rfc_clean) >= 10:
                # Los primeros 10 caracteres de CURP y RFC (Iniciales + Fecha YYMMDD) deben coincidir
                if curp_clean[:10] != rfc_clean[:10]:
                    curp_rfc_match = False
                    detalles.append(
                        f"Inconsistencia de identidad: CURP ({curp_clean[:10]}) y RFC ({rfc_clean[:10]}) no coinciden en sus primeros 10 caracteres."
                    )
            else:
                detalles.append("CURP o RFC demasiado cortos para validar consistencia.")

        return {
            "poliza_vs_ine_nombre_match": nombre_match,
            "curp_rfc_consistent": curp_rfc_match,
            "detalles": detalles,
        }

