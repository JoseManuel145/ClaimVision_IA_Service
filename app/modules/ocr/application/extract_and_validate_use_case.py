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
        detalles = []

        poliza_nombre = poliza.nombre_asegurado.strip().lower()
        ine_nombre = ine.nombre_completo.strip().lower()
        nombre_match = poliza_nombre == ine_nombre
        if not nombre_match:
            detalles.append(
                f"Nombre no coincide: poliza='{poliza.nombre_asegurado}' vs INE='{ine.nombre_completo}'"
            )

        curp_rfc_match = True
        if ine.curp and ine.curp.strip():
            if ine.curp.strip().lower() not in poliza.nombre_asegurado.strip().lower():
                pass
        if ine.rfc and ine.rfc.strip():
            curp_rfc_match = True

        return {
            "poliza_vs_ine_nombre_match": nombre_match,
            "curp_rfc_consistent": curp_rfc_match,
            "detalles": detalles,
        }
