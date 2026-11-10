# Pokémon Detector

A regular Discord bot that checks image attachments and embed images in messages it can access, compares them with a locally cached reference dataset from PokeAPI, and replies in the same channel when a match is sufficiently confident.

## Features
- Automatic reference image collection from PokeAPI (no manual image uploads required).
- Perceptual image matching using pHash, dHash and wavelet hash.
- Conservative ambiguity check to reduce confident wrong guesses.
- Checks message attachments, embed images and thumbnails.
- Bounded background image processing so commands are not held up by image downloads.
- Owner-only presence status and voice join/leave commands.
- Feedback forwarded to the owner by DM; owner can reply directly to the forwarded DM to relay a response.
- SQLite cache with WAL mode, suitable for a persistent server disk.
- Normal Discord bot account; no self-bot automation.

## Requirements
- Python 3.10 or newer recommended.
- A Discord application and bot token.
- Message Content Intent enabled in the Discord Developer Portal.
- Bot permissions: View Channels, Read Message History, Send Messages, Embed Links; Connect and Speak if voice commands are needed.

## Run locally or on a Linux VPS
1. Clone the repository:
   ```bash
   git clone https://github.com/DevRedant/Pokemon-Detector.git
   cd Pokemon-Detector
   ```
2. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Set environment variables using your hosting provider's secret/environment-variable settings. Copy `.env.example` to `.env` only for local use.
4. Start the bot:
   ```bash
   python bot.py
   ```

## Oracle Cloud deployment
- Use an Ubuntu VM and install Python 3, pip and the dependencies in `requirements.txt`.
- Clone this repository and set the environment variables in a private environment file outside the repository or in the service manager.
- Keep the SQLite database on the VM's persistent boot/block volume. Do not store it on temporary storage.
- Run with a systemd service so it starts on boot and restarts after crashes. Set `WorkingDirectory` to the cloned repository and `ExecStart` to the virtual environment's Python executable followed by `bot.py`.
- Never commit `.env`, bot tokens, or credentials.

## Environment variables
| Variable | Default | Meaning |
|---|---:|---|
| DISCORD_TOKEN | required | Bot token |
| OWNER_ID | required | Your numeric Discord user ID |
| PREFIX | `!` | Command prefix |
| MATCH_THRESHOLD | `14` | Maximum average hash distance |
| MATCH_MIN_MARGIN | `1.5` | Minimum lead over a different Pokémon |
| MAX_IMAGE_SIZE | `8388608` | Maximum downloaded image bytes |
| DATABASE | `data/pokemon.db` | SQLite database path |
| DATASET_CONCURRENCY | `8` | Concurrent dataset workers |
| DETECTION_CONCURRENCY | `3` | Concurrent message detections |

## Commands
- `!help`
- `!feedback your message`
- `!dataset`
- `!dataset sync` (owner only)
- `!status online|idle|dnd|invisible` (owner only)
- `!join` and `!leave` (owner only)

For feedback replies, reply directly to the bot's forwarded feedback message in your DM with the bot.

## Important limitations
- This is perceptual image matching, not a trained vision model. It will not reliably identify every Pokémon, shiny, regional form, Mega form, Gigantamax form, or stylized spawn image.
- PokeAPI only supplies images it exposes for each API record. Some alternate forms have no usable sprite/artwork.
- Accuracy depends on source image similarity. Heavy crops, text overlays, custom artwork and animation frames may prevent a match.
- The bot only sees channels where it has View Channel and Read Message History permissions. Message Content Intent must be enabled in the Discord Developer Portal and in this code.
- Keep the database on persistent storage. If the host deletes its disk, the cache must be rebuilt.
