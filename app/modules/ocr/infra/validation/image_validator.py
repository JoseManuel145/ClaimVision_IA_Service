import io
from io import BytesIO
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image

from app.modules.ocr.infra.validation.models import ValidationResult


class DocumentType(str, Enum):
    INE = "ine"
    POLIZA = "poliza"


class ImageValidator:

    INE_ASPECT_RATIOS = [0.631, 1.585]
    INE_MIN_WIDTH = 800
    INE_MIN_HEIGHT = 500
    INE_ASPECT_TOLERANCE = 0.25
    INE_MIN_SHARPNESS = 50.0
    INE_BRIGHTNESS_MIN = 50.0
    INE_BRIGHTNESS_MAX = 220.0

    POLIZA_ASPECT_RATIOS = [0.773, 1.294]
    POLIZA_MIN_WIDTH = 1000
    POLIZA_MIN_HEIGHT = 700
    POLIZA_ASPECT_TOLERANCE = 0.25
    POLIZA_MIN_SHARPNESS = 30.0
    POLIZA_BRIGHTNESS_MIN = 60.0
    POLIZA_BRIGHTNESS_MAX = 220.0

    async def validate_ine_image(self, file_bytes: bytes, content_type: str) -> ValidationResult:
        if content_type == "application/pdf":
            return await self._validate_ine_pdf(file_bytes)
        return await self._validate_image(file_bytes, DocumentType.INE)

    async def validate_poliza_pdf(self, file_bytes: bytes) -> ValidationResult:
        return await self._validate_pdf(file_bytes, DocumentType.POLIZA)

    async def _validate_ine_pdf(self, pdf_bytes: bytes) -> ValidationResult:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                doc.close()
                return self._fail_result("El PDF no contiene paginas")

            worst_result = None
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                result = self._validate_image_data(img, DocumentType.INE)
                if worst_result is None or len(result.errors) > len(worst_result.errors):
                    worst_result = result

            doc.close()
            return worst_result or self._fail_result("No se pudo procesar el PDF")
        except Exception as e:
            return self._fail_result(f"Error al procesar PDF: {str(e)}")

    async def _validate_pdf(self, pdf_bytes: bytes, doc_type: DocumentType) -> ValidationResult:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(doc) == 0:
                doc.close()
                return self._fail_result("El PDF no contiene paginas")

            worst_result = None
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                result = self._validate_image_data(img, doc_type)
                if worst_result is None or len(result.errors) > len(worst_result.errors):
                    worst_result = result

            doc.close()
            return worst_result or self._fail_result("No se pudo procesar el PDF")
        except Exception as e:
            return self._fail_result(f"Error al procesar PDF: {str(e)}")

    async def _validate_image(self, file_bytes: bytes, doc_type: DocumentType) -> ValidationResult:
        try:
            img = Image.open(BytesIO(file_bytes))
            return self._validate_image_data(img, doc_type)
        except Exception as e:
            return self._fail_result(f"Error al abrir imagen: {str(e)}")

    def _validate_image_data(self, img: Image.Image, doc_type: DocumentType) -> ValidationResult:
        if doc_type == DocumentType.INE:
            min_w, min_h = self.INE_MIN_WIDTH, self.INE_MIN_HEIGHT
            target_ratios = self.INE_ASPECT_RATIOS
            tolerance = self.INE_ASPECT_TOLERANCE
            min_sharpness = self.INE_MIN_SHARPNESS
            brightness_min = self.INE_BRIGHTNESS_MIN
            brightness_max = self.INE_BRIGHTNESS_MAX
        else:
            min_w, min_h = self.POLIZA_MIN_WIDTH, self.POLIZA_MIN_HEIGHT
            target_ratios = self.POLIZA_ASPECT_RATIOS
            tolerance = self.POLIZA_ASPECT_TOLERANCE
            min_sharpness = self.POLIZA_MIN_SHARPNESS
            brightness_min = self.POLIZA_BRIGHTNESS_MIN
            brightness_max = self.POLIZA_BRIGHTNESS_MAX

        result = ValidationResult(is_valid=True)
        width, height = img.size
        result.image_width = width
        result.image_height = height

        if width < min_w or height < min_h:
            result.add_error(
                f"Resolucion insuficiente: {width}x{height} (minimo {min_w}x{min_h})"
            )

        aspect = width / height if height > 0 else 0
        result.aspect_ratio = aspect
        if not self._check_aspect_ratio(aspect, target_ratios, tolerance):
            ratios_str = " o ".join([f"~{r:.2f}" for r in target_ratios])
            result.add_warning(
                f"Aspecto no estandar: {aspect:.2f} (estandar {ratios_str})"
            )

        gray = img.convert("L")
        sharpness = self._calculate_sharpness(gray)
        result.sharpness = sharpness
        if sharpness < min_sharpness:
            result.add_error(
                f"Imagen borrosa (nitidez: {sharpness:.1f}, minima: {min_sharpness})"
            )

        brightness = self._calculate_brightness(gray)
        result.brightness_mean = brightness
        if brightness < brightness_min:
            result.add_error(
                f"Imagen oscura (brillo: {brightness:.1f}, minimo: {brightness_min})"
            )
        elif brightness > brightness_max:
            result.add_error(
                f"Imagen sobreexpuesta (brillo: {brightness:.1f}, maximo: {brightness_max})"
            )

        return result

    def _check_aspect_ratio(self, ratio: float, targets: list[float], tolerance: float) -> bool:
        for target in targets:
            if abs(ratio - target) / target <= tolerance:
                return True
        return False

    def _calculate_sharpness(self, gray: Image.Image) -> float:
        img_array = np.array(gray, dtype=np.float64)
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        from scipy.ndimage import convolve
        laplacian_result = convolve(img_array, laplacian)
        return float(np.var(laplacian_result))

    def _calculate_brightness(self, gray: Image.Image) -> float:
        img_array = np.array(gray, dtype=np.float64)
        return float(np.mean(img_array))

    def _fail_result(self, message: str) -> ValidationResult:
        result = ValidationResult(is_valid=False)
        result.add_error(message)
        return result
