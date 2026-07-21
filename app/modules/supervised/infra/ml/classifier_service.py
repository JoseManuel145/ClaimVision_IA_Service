import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


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


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


def _load_class_names(mapping_path: Path) -> list[str]:
    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
        return [mapping[str(i)] for i in range(len(mapping))]
    return []


class ResNetClassifierService:
    def __init__(self, models_dir: str):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models_dir = Path(models_dir)
        self._model_path = self._models_dir / "classifier_best.pth"
        self._mapping_path = self._models_dir / "class_mapping.json"
        self._class_names = _load_class_names(self._mapping_path)
        self._load_model()

    def _load_model(self) -> None:
        num_classes = max(len(self._class_names), 1)
        self._model = _build_model(num_classes)
        if self._model_path.exists():
            state = torch.load(str(self._model_path), map_location=self._device, weights_only=True)
            actual_num = state["fc.weight"].size(0)
            if actual_num != num_classes:
                self._class_names = [f"Clase_{i}" for i in range(actual_num)]
                num_classes = actual_num
                self._model = _build_model(num_classes)
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
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImageFolderDataset(data_dir, train_transform)
        if len(dataset) == 0:
            raise ValueError("No se encontraron imágenes válidas en los datos enviados")

        all_labels = [s[1] for s in dataset.samples]
        num_classes = max(all_labels) + 1

        n_samples = len(dataset)
        n_val = int(n_samples * 0.2)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(n_samples), all_labels))

        val_dataset_raw = ImageFolderDataset(data_dir, val_transform)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(val_dataset_raw, val_idx)

        loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        train_labels = [all_labels[i] for i in train_idx]
        train_label_counts = Counter(train_labels)
        weight_tensor = torch.ones(num_classes, dtype=torch.float32)
        total_train = len(train_labels)
        for cls_id in range(num_classes):
            if cls_id in train_label_counts and train_label_counts[cls_id] > 0:
                weight_tensor[cls_id] = total_train / (num_classes * train_label_counts[cls_id])
        weight_tensor = weight_tensor.to(device)

        model = _build_model(num_classes)
        model.train()
        model.to(device)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
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

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            scheduler.step(epoch_loss)

            is_best = False
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1

            on_epoch_end(job_id, epoch + 1, val_acc, epoch_loss, is_best)

            if is_best:
                temp_path = str(self._model_path) + ".tmp"
                torch.save(model.state_dict(), temp_path)
                os.replace(temp_path, str(self._model_path))

            if patience_counter >= 5:
                break

        self._class_names = [f"Clase_{i}" for i in range(num_classes)]
        mapping = {str(i): self._class_names[i] for i in range(num_classes)}
        with open(self._mapping_path, "w") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        self._load_model()

    def get_class_names(self) -> list[str]:
        return self._class_names

    def get_severity(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "Bajo"
        elif confidence >= 0.5:
            return "Medio"
        return "Alto"
