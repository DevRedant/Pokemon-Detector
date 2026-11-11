"""Automatically download Pokémon reference art and train the local classifier.

Run on a separate PC with adequate RAM:
    python -m pip install -r requirements-train.txt
    python train_ai.py --epochs 12

The Discord bot does not call an AI API. PokeAPI is used only as an image catalogue.
"""
import argparse
import asyncio
import io
import json
import logging
from pathlib import Path

import aiohttp
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

API = "https://pokeapi.co/api/v2"
log = logging.getLogger("pokemon-training")


async def download_dataset(root: Path, concurrency: int = 8):
    """Fetch image assets and use the API record name as their automatic label."""
    root.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=45, connect=15)
    connector = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)
    headers = {"User-Agent": "PokemonDetectorLocalTrainer/1.0"}
    semaphore = asyncio.Semaphore(concurrency)
    saved = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers) as session:
        async with session.get(f"{API}/pokemon?limit=2000") as response:
            response.raise_for_status()
            entries = (await response.json()).get("results", [])
        print(f"Found {len(entries)} Pokémon/form records. Downloading image assets...")

        async def process(entry):
            nonlocal saved
            async with semaphore:
                try:
                    async with session.get(entry["url"]) as response:
                        response.raise_for_status()
                        detail = await response.json()
                    sprites = detail.get("sprites") or {}
                    urls = []
                    for key in ("front_default", "front_shiny"):
                        if sprites.get(key):
                            urls.append(sprites[key])
                    other = sprites.get("other") or {}
                    for group in ("official-artwork", "home", "showdown"):
                        art = other.get(group) or {}
                        for key in ("front_default", "front_shiny"):
                            if art.get(key):
                                urls.append(art[key])
                    label = (detail.get("name") or entry["name"]).lower().replace("-", "_")
                    folder = root / label
                    folder.mkdir(parents=True, exist_ok=True)
                    for index, url in enumerate(dict.fromkeys(urls)):
                        try:
                            async with session.get(url) as image_response:
                                if image_response.status != 200:
                                    continue
                                raw = await image_response.content.read(8 * 1024 * 1024 + 1)
                            if len(raw) > 8 * 1024 * 1024:
                                continue
                            with Image.open(io.BytesIO(raw)) as source:
                                image = source.convert("RGB")
                                image.thumbnail((1000, 1000))
                                destination = folder / f"{index}.jpg"
                                image.save(destination, "JPEG", quality=92)
                                saved += 1
                        except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
                            continue
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as exc:
                    log.warning("Skipped %s: %s", entry.get("name"), exc)

        await asyncio.gather(*(process(entry) for entry in entries))
    print(f"Downloaded {saved} labelled images into {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/auto_training")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--no-download", action="store_true",
                        help="Use an existing dataset instead of downloading from PokeAPI")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        asyncio.run(download_dataset(data_path))

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
    train_all = datasets.ImageFolder(str(data_path), transform=train_transform)
    eval_all = datasets.ImageFolder(str(data_path), transform=eval_transform)
    if len(train_all.classes) < 2:
        raise SystemExit("Not enough Pokémon image classes were downloaded.")

    # Keep only classes with at least two examples so every class can be evaluated.
    counts = {}
    for _, class_index in train_all.samples:
        counts[class_index] = counts.get(class_index, 0) + 1
    kept = [i for i, count in counts.items() if count >= 2]
    if len(kept) < 2:
        raise SystemExit("Not enough classes have two or more images.")
    indices_by_class = {i: [] for i in kept}
    for index, (_, class_index) in enumerate(train_all.samples):
        if class_index in indices_by_class:
            indices_by_class[class_index].append(index)
    train_indices, val_indices = [], []
    for class_indices in indices_by_class.values():
        class_indices.sort()
        split = max(1, int(len(class_indices) * 0.8))
        if split >= len(class_indices):
            split = len(class_indices) - 1
        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])

    # Rebuild a compact class mapping after excluding single-image classes.
    kept_names = [train_all.classes[i] for i in kept]
    remap = {old: new for new, old in enumerate(kept)}
    train_samples = [(path, remap[label]) for path, label in train_all.samples if label in remap]
    train_all.samples = train_samples
    train_all.imgs = train_samples
    eval_all.samples = train_samples
    eval_all.imgs = train_samples
    train_loader = DataLoader(Subset(train_all, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(eval_all, val_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    print(f"Training {len(kept_names)} Pokémon classes using {len(train_indices)} images; "
          f"validation images: {len(val_indices)}")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(kept_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predicted = model(images.to(device)).argmax(1).cpu()
                correct += (predicted == labels).sum().item()
                total += labels.numel()
        accuracy = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{args.epochs} loss={loss_sum/max(1,len(train_loader)):.4f} "
              f"val_accuracy={accuracy:.3f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), output_path / "pokemon_mobilenet.pth")

    model.load_state_dict(torch.load(output_path / "pokemon_mobilenet.pth", map_location=device))
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224),
                      output_path / "pokemon_mobilenet.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=12)
    (output_path / "pokemon_labels.json").write_text(
        json.dumps({"classes": kept_names}, indent=2), encoding="utf-8")
    print(f"Exported ONNX model with {len(kept_names)} Pokémon classes to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
