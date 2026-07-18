import re
import io
from PIL import Image, ImageEnhance, ImageFilter
import fitz
import pytesseract


class PyMuPDFOCRService:
    MIN_TEXT_LENGTH = 20
    _MRZ_PATTERN = re.compile(r"[A-Z]{3,}<[A-Z]{3,}<<[A-Z]{3,}")

    async def extract(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Pass 1: try direct text extraction
        text_parts = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                text_parts.append(text)
        full_text = "\n".join(text_parts)
        if len(full_text) >= self.MIN_TEXT_LENGTH:
            doc.close()
            return full_text

        # Pass 2: extract embedded images, try multiple OCR methods, pick best
        ocr_parts = []
        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(image_bytes))
                    text = self._ocr_best_method(img)
                    if text.strip():
                        ocr_parts.append(text.strip())
                except Exception:
                    continue

        if ocr_parts:
            doc.close()
            return "\n".join(ocr_parts)

        # Pass 3: render pages as images and OCR
        text_parts.clear()
        for page in doc:
            pix = page.get_pixmap(dpi=400)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = self._ocr_best_method(img)
            text_parts.append(text.strip())
        doc.close()
        return "\n".join(text_parts)

    def _ocr_best_method(self, img: Image.Image) -> str:
        text_simple = self._ocr_simple(img)
        text_enhanced = self._ocr_enhanced(img)
        simple_has_mrz = bool(self._MRZ_PATTERN.search(text_simple.upper()))
        enhanced_has_mrz = bool(self._MRZ_PATTERN.search(text_enhanced.upper()))
        if simple_has_mrz and not enhanced_has_mrz:
            return text_simple
        if len(text_enhanced) > len(text_simple) * 1.2:
            return text_enhanced
        return text_simple

    def _ocr_simple(self, img: Image.Image) -> str:
        gray = img.convert("L")
        return pytesseract.image_to_string(gray, lang="spa+eng", config="--psm 6")

    def _ocr_enhanced(self, img: Image.Image) -> str:
        upscale = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        gray = upscale.convert("L")
        sharp = gray.filter(ImageFilter.SHARPEN)
        enhanced = ImageEnhance.Contrast(sharp).enhance(2.0)
        bw = enhanced.point(lambda x: 0 if x < 128 else 255, "1")
        return pytesseract.image_to_string(bw, lang="spa+eng", config="--psm 6")
