import io
from PIL import Image
import torch
from torchvision import transforms


class SupervisedPreprocessor:
    def __init__(self, img_size: int = 224):
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    async def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(image)
        tensor = tensor.unsqueeze(0)
        return tensor
