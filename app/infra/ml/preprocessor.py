import io
from PIL import Image
import torch
from torchvision import transforms
from app.domain.ports import ImagePreprocessor


class TorchImagePreprocessor(ImagePreprocessor):
    def __init__(self, img_size: int = 224):
        self._transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    async def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor
