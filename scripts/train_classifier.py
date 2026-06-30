"""
Script de entrenamiento offline del clasificador supervisado (ResNet18).

Uso:
    python scripts/train_classifier.py \
        --data-dir /ruta/images \
        --labels /ruta/train.csv \
        --output-dir ./models \
        --num-classes 6 \
        --label-map '{"1":0,"2":1,"3":2,"4":3,"5":4,"6":5}'

Formato esperado de labels.csv:
    filename,label
    img001.jpg,2
    img002.jpg,4
    ...

--label-map mapea los valores del CSV a las clases del modelo (0-based).
--class-names lista los nombres de cada clase (opcional, por defecto: Clase_0...).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image


class MappedCSVDataset(Dataset):
    def __init__(self, data_dir: str, csv_path: str, label_map: dict, transform):
        self.samples = []
        self.transform = transform
        self.label_map = label_map
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = int(row["label"])
                if raw not in self.label_map:
                    continue
                mapped = self.label_map[raw]
                img_path = Path(data_dir) / row["filename"]
                if img_path.exists():
                    self.samples.append((str(img_path), mapped))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    try:
        raw_map = json.loads(args.label_map) if args.label_map else None
        label_map = {int(k): v for k, v in raw_map.items()} if raw_map else None
    except json.JSONDecodeError:
        print("Error: --label-map no es un JSON válido", file=sys.stderr)
        sys.exit(1)

    if label_map:
        print(f"Mapeo de etiquetas: {label_map}")
        num_classes = max(label_map.values()) + 1
        if args.num_classes and args.num_classes != num_classes:
            print(f"Advertencia: --num-classes ({args.num_classes}) no coincide con el máximo del mapeo ({num_classes}), usando {num_classes}")
    else:
        label_map = {}
        num_classes = args.num_classes
        print(f"Usando etiquetas del CSV tal cual, {num_classes} clases")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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

    full_dataset = MappedCSVDataset(args.data_dir, args.labels, label_map, train_transform)
    if len(full_dataset) == 0:
        print("Error: no se encontraron imágenes válidas", file=sys.stderr)
        sys.exit(1)
    print(f"Total de muestras: {len(full_dataset)}")

    labels_set = sorted(set(s[1] for s in full_dataset.samples))
    print(f"Clases presentes: {labels_set}")

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes)
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0.0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.class_names:
        class_names = json.loads(args.class_names)
    else:
        class_names = [f"Clase_{i}" for i in range(num_classes)]

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
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

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d}/{args.epochs}  "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}  "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model_path = output_dir / "classifier_best.pth"
            torch.save(model.state_dict(), str(model_path))
            mapping = {str(i): class_names[i] for i in range(num_classes)}
            with open(output_dir / "class_mapping.json", "w") as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
            print(f"  → Nuevo mejor modelo guardado (acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping en época {epoch+1}")
                break

    print(f"\nEntrenamiento completado. Mejor accuracy en validación: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar clasificador de daños vehiculares")
    parser.add_argument("--data-dir", required=True, help="Directorio con las imágenes")
    parser.add_argument("--labels", required=True, help="CSV con filename,label")
    parser.add_argument("--output-dir", default="models", help="Directorio de salida")
    parser.add_argument("--num-classes", type=int, default=7, help="Número de clases")
    parser.add_argument("--label-map", help='JSON: mapeo de etiquetas CSV a clases. Ej: \'{"1":0,"2":1}\'')
    parser.add_argument("--class-names", help='JSON: nombres de cada clase. Ej: \'["Sin Daño","Rotura"]\'')
    parser.add_argument("--epochs", type=int, default=50, help="Cantidad de épocas")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--early-stop", type=int, default=5, help="Paciencia para early stopping")
    args = parser.parse_args()
    train(args)
