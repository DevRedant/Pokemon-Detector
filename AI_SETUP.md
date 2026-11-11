# Local AI setup

The project can automatically download Pokémon images and labels from PokeAPI, then train and export a local MobileNetV3 classifier. The running Discord bot uses the exported ONNX model locally and does not call an AI inference API.

## 1. Train automatically (on a separate PC)

Training needs more memory and compute than the bot's inference process. On a PC with Python 3.10 or 3.11:

```bash
pip install -r requirements-train.txt
python train_ai.py --epochs 12
```

The script automatically:
- Retrieves Pokémon/form records from PokeAPI
- Downloads available normal, shiny, official-artwork, HOME and Showdown images
- Uses each API record's Pokémon name as its label
- Skips classes that have fewer than two downloaded images
- Fine-tunes a MobileNetV3-Small classifier head and exports the model

No manual `.learn` commands or hand-labelled image folders are required for this automatic sprite/artwork dataset. You can use `--no-download` to train from an existing `data/auto_training` folder.

## 2. Copy the exported model to the bot host

After training, copy these two files from the training PC into the bot's `models/` directory:

- `models/pokemon_mobilenet.onnx`
- `models/pokemon_labels.json`

Restart the bot and run `.ai` to check that the files are detected. Keep the model files on persistent storage.

## Important accuracy limitation

PokeAPI provides Pokémon sprites and artwork, not Pokétwo's actual spawn screenshots. This automatically trained model can classify images resembling its training artwork, but it is **not guaranteed to recognise real spawn scenes with backgrounds, effects, overlays or unusual crops**. Synthetic backgrounds alone cannot reliably reproduce Pokétwo's rendering. Strong scene-level accuracy would require a suitable collection of real spawn examples; the bot does not fabricate those labels.

The model is a whole-image classifier, not an object detector. It predicts one Pokémon from the image crop. Treat predictions as guesses and validate with unseen images before relying on them.

## Commands

- `.ai` — show local model status and collected optional examples.
- `.learn <Pokemon>` — optional owner-only manual scene-example collector; not required for automatic sprite training.
- `.dataset` — show the existing perceptual-hash reference image count.
