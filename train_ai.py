"""Train a local MobileNetV3 Pokémon classifier from owner-labelled spawn images.

Collect examples with .learn <Pokémon name> by replying to a spawn image, then run:
    pip install -r requirements-train.txt
    python train_ai.py --epochs 12

Training runs separately from the Discord bot. It needs more RAM than inference.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise SystemExit("No training folder found. Collect images with .learn first.")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # ImageFolder class names are folder names created by the .learn command.
    training_view = datasets.ImageFolder(str(data_path), transform=train_transform)
    evaluation_view = datasets.ImageFolder(str(data_path), transform=eval_transform)
    classes = training_view.classes
    if len(classes) < 2:
        raise SystemExit("At least two Pokémon classes are required to train a classifier.")
    counts = {name: len(list((data_path / name).glob("*"))) for name in classes}
    if any(count < 2 for count in counts.values()):
        raise SystemExit("Collect at least 2 labelled images for every Pokémon class.")
    if len(training_view) < 20:
        raise SystemExit("Collect at least 20 labelled spawn images before training.")

    # Use ImageNet transfer learning; weights are downloaded by torchvision once
    # during training. The running Discord bot makes no external model/API calls.
    train_size = max(1, int(len(training_view) * 0.9))
    val_size = len(training_view) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(training_view, [train_size, val_size], generator=generator)
    # Apply deterministic transforms to validation samples.
    val_subset.dataset = evaluation_view
    workers = 0
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                              num_workers=workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers)

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                correct += (outputs.argmax(1).cpu() == labels).sum().item()
                total += labels.numel()
        accuracy = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{args.epochs} loss={running_loss/max(1,len(train_loader)):.4f} val_accuracy={accuracy:.3f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), output_path / "pokemon_mobilenet.pth")

    model.load_state_dict(torch.load(output_path / "pokemon_mobilenet.pth", map_location=device))
    model.eval().cpu()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, dummy, output_path / "pokemon_mobilenet.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=12)
    (output_path / "pokemon_labels.json").write_text(
        json.dumps({"classes": classes}, indent=2), encoding="utf-8"
    )
    print(f"Exported ONNX model with {len(classes)} classes to {output_path}")


if __name__ == "__main__":
    main()
