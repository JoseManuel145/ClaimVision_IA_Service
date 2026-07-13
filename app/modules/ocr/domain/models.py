from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OCRDocument:
    id: str
    filename: str
    text: str
    page_count: int
    created_at: datetime


@dataclass
class PolizaData:
    numero_poliza: str
    aseguradora: str
    nombre_asegurado: str
    vehiculo_marca: str
    vehiculo_modelo: str
    vehiculo_anio: int
    vehiculo_placas: str
    vehiculo_vin: str | None
    vehiculo_color: str | None
    vigencia_inicio: str
    vigencia_fin: str


@dataclass
class IneData:
    nombre_completo: str
    curp: str
    rfc: str | None
    fecha_nacimiento: str
    sexo: str
    domicilio: str
    clave_elector: str
    numero_credencial: str | None


@dataclass
class DocumentExtraction:
    id: str
    filename: str
    document_type: str
    raw_text: str
    extracted_data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now())
