import io
from PIL import Image
import imagehash


class ImageHasher:
    def __init__(self, threshold: int = 5):
        self._threshold = threshold

    def compute_hash(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return str(imagehash.phash(image))

    def are_duplicates(self, hash1: str, hash2: str) -> bool:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2 <= self._threshold
