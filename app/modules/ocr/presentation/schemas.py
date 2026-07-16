from pydantic import BaseModel
from typing import Optional


class OcrResponse(BaseModel):
    id: str
    filename: str
    text: str
    page_count: int
    created_at: str


class OcrHistoryItem(BaseModel):
    id: str
    filename: str
    page_count: int
    created_at: str


class OcrHistoryResponse(BaseModel):
    data: list[OcrHistoryItem]
    total: int
    page: int
    limit: int


class PolizaExtractedResponse(BaseModel):
    id: str
    filename: str
    numero_poliza: str
    aseguradora: str
    nombre_asegurado: str
    vehiculo_marca: str
    vehiculo_modelo: str
    vehiculo_anio: int
    vehiculo_placas: str
    vehiculo_vin: Optional[str] = None
    vehiculo_color: Optional[str] = None
    vigencia_inicio: str
    vigencia_fin: str


class IneExtractedResponse(BaseModel):
    id: str
    filename: str
    nombre_completo: str
    curp: str
    rfc: Optional[str] = None
    fecha_nacimiento: Optional[str] = None
    sexo: Optional[str] = None
    domicilio: Optional[str] = None
    clave_elector: Optional[str] = None


class ValidationResult(BaseModel):
    poliza_vs_ine_nombre_match: bool
    curp_rfc_consistent: bool
    detalles: list[str]


class ExtractAndValidateResponse(BaseModel):
    poliza: PolizaExtractedResponse
    ine: IneExtractedResponse
    validation: ValidationResult
