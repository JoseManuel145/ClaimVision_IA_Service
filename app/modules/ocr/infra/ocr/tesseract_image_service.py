from io import BytesIO
from PIL import Image
import pytesseract


class TesseractImageOCRService:
    async def extract_from_image(self, image_bytes: bytes, filename: str) -> str:
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang="spa+eng")
        return text.strip()
