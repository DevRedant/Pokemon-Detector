# Local AI setup

This project supports optional **local-only** Pokémon image classification with ONNX Runtime.
The Discord bot does not call a third-party AI API. The model is loaded from local files.

## 1. Collect real Pokétwo spawn examples

1. Start the bot and make sure it can see the spawn image.
2. Reply directly to that spawn image with `.learn Pikachu` (replace Pikachu with the correct label).
3. Repeat for many Pokémon and varied backgrounds. The command is owner-only.
4. Images are stored locally under `data/training/<pokemon-name>/`.

Use correctly labelled images. Do not train on guessed labels. Aim for at least 20-50 varied examples per Pokémon to start; more is better. The classifier is only as good as the labelled scene examples.

## 2. Train away from the bot host

Training needs considerably more memory than inference. On a PC with Python 3.10 or 3.11:

```bash
pip install -r requirements-train.txt
python train_ai.py --epochs 12
```

The training script uses torchvision's pretrained MobileNetV3-Small ImageNet weights as a starting point. This is a downloadable open-source model weight, **not an AI API**. Training and export happen locally on that machine. Copy these two generated files into the repository/deployment:

- `models/pokemon_mobilenet.onnx`
- `models/pokemon_labels.json`

Do not commit private images or large model files to a public repository. Upload them to your hosting's persistent storage instead, and set `AI_MODEL_PATH` and `AI_LABELS_PATH` if needed.

## 3. Install runtime dependencies and restart

```bash
pip install -r requirements.txt
```

The bot tries hash matching first, then local AI if a model is present. If no model is present or confidence is below `AI_CONFIDENCE`, it reports no match rather than inventing a Pokémon.

## Commands

- `.ai` — owner-only model status and number of collected examples.
- `.learn <Pokemon>` — owner-only; reply to a spawn image to save a labelled training example.
- `.dataset` — existing reference image count.

## Important limitations

- The existing PokeAPI sprite database is not a scene-image training set.
- A useful model requires correctly labelled Pokétwo-style spawn screenshots, ideally with varied backgrounds, sizes, and visual effects.
- This is image classification, not object detection. It predicts from a centered crop; screenshots with the Pokémon far from the center may be missed.
- Validation accuracy on a small dataset is not proof of real-world accuracy. Test with unseen spawn screenshots.
