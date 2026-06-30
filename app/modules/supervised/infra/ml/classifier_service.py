import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = [
    "Sin Daño",
    "Rotura_Cristal",
    "Rayadura",
    "Abolladura",
    "Grietas",
    "Neumático_pinchado",
    "Faro_roto",
]


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir: str, transform):
        self.samples = []
        self.transform = transform
        labels_path = Path(data_dir) / "labels.json"
        with open(labels_path) as f:
            entries = json.load(f)
        for entry in entries:
            img_path = Path(data_dir) / entry["filename"]
            if img_path.exists():
                self.samples.append((str(img_path), entry["label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def _build_model(num_classes: int = 7) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


class ResNetClassifierService:
    def __init__(self, models_dir: str):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_dir = Path(models_dir)
        self._model_path = self._models_dir / "classifier_best.pth"
        self._mapping_path = self._models_dir / "class_mapping.json"
        self._load_model()

    def _load_model(self) -> None:
        self._model = _build_model()
        if self._model_path.exists():
            state = torch.load(str(self._model_path), map_location=self._device, weights_only=True)
            self._model.load_state_dict(state, strict=True)
        self._model.eval()
        self._model.to(self._device)

    async def predict(self, tensor: torch.Tensor) -> tuple[int, float, list[float]]:
        with torch.no_grad():
            output = self._model(tensor.to(self._device))
            probs = torch.softmax(output, dim=1)
            confidence, class_id = torch.max(probs, dim=1)
        prob_dist = probs.squeeze(0).tolist()
        return int(class_id.item()), float(confidence.item()), prob_dist

    def retrain(
        self,
        data_dir: str,
        epochs: int,
        lr: float,
        job_id: str,
        on_epoch_end: callable,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolderDataset(data_dir, train_transform)
        if len(dataset) == 0:
            raise ValueError("No se encontraron imágenes válidas en los datos enviados")
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        model = _build_model()
        model.train()
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            scheduler.step(epoch_loss)

            is_best = False
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1

            on_epoch_end(job_id, epoch + 1, epoch_acc, epoch_loss, is_best)

            if is_best:
                temp_path = str(self._model_path) + ".tmp"
                torch.save(model.state_dict(), temp_path)
                os.replace(temp_path, str(self._model_path))
                mapping = {str(i): name for i, name in enumerate(CLASS_NAMES)}
                with open(self._mapping_path, "w") as f:
                    json.dump(mapping, f, indent=2, ensure_ascii=False)

            if patience_counter >= 5:
                break

        self._load_model()

    def get_class_names(self) -> list[str]:
        return CLASS_NAMES

    def get_severity(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "Bajo"
        elif confidence >= 0.5:
            return "Medio"
        return "Alto"
