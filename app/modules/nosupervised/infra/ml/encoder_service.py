import json
import torch
import torch.nn as nn
from pathlib import Path
from app.modules.nosupervised.domain.ports import EncoderService


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128, channels: tuple = (32, 64, 128, 256), img_size: int = 224):
        super().__init__()
        modules = []
        in_c = 3
        for out_c in channels:
            modules.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ])
            in_c = out_c
        self.conv = nn.Sequential(*modules)
        h = img_size // (2 ** len(channels))
        flat_dim = channels[-1] * h * h
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TorchEncoderService(EncoderService):
    def __init__(self, models_dir: str):
        self._device = torch.device("cpu")
        config_path = Path(models_dir) / "encoder_config.json"
        weights_path = Path(models_dir) / "encoder_best.pth"

        with open(config_path) as f:
            cfg = json.load(f)

        self._encoder = Encoder(
            latent_dim=cfg["latent_dim"],
            img_size=cfg["img_size"],
        )
        state = torch.load(str(weights_path), map_location=self._device, weights_only=True)
        self._encoder.load_state_dict(state, strict=False)
        self._encoder.eval()
        self._encoder.to(self._device)

    async def encode(self, tensor: torch.Tensor) -> list[float]:
        with torch.no_grad():
            vector = self._encoder.forward(tensor.to(self._device))
        return vector.squeeze(0).tolist()
