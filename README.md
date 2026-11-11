# Pokémon Detector

A Discord bot that automatically identifies Pokémon in image attachments and embeds with a locally running AI model.

## Setup

1. Put your Discord bot token in `config.py`. Do not upload your real token to GitHub.
2. Install dependencies: `pip install -r requirements.txt`
3. Start: `python bot.py`

The first startup downloads the pretrained model automatically. No manual Pokémon image collection, labelling, or training is required.

## Commands

- `.help` — command guide
- `.status` — bot and AI model status

## Important hosting note

The pretrained 1,027-class Vision Transformer is roughly 350 MB of model weights and needs additional RAM for Python, PyTorch, image processing, and inference. A 600 MB RAM container may run out of memory. Use a host with at least 2 GB RAM for a reliable deployment. The model runs locally; no inference API is used.

Model: https://huggingface.co/imzynoxprince/pokemons-image-classifier-gen1-gen9
