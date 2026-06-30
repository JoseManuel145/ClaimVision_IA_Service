from PIL import Image
import fitz
import pytesseract


class PyMuPDFOCRService:
    MIN_TEXT_LENGTH = 20

    async def extract(self, pdf_bytes: bytes) -> str:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                text_parts.append(text)
        full_text = "\n".join(text_parts)
        if len(full_text) >= self.MIN_TEXT_LENGTH:
            doc.close()
            return full_text
        text_parts.clear()
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang="spa+eng")
            text_parts.append(text.strip())
        doc.close()
        return "\n".join(text_parts)
