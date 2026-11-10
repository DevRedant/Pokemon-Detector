# Pokémon Detector (starter v1)

A regular Discord bot that scans image attachments and embed images in guild messages, compares them with a locally cached PokeAPI reference dataset, and replies in the same channel when a match is sufficiently close.

## Important limitations
- This is a starter version, not a promise of perfect recognition.
- It matches perceptual hashes against images supplied by PokeAPI. Images from third-party spawn bots may be cropped, stylized, compressed, or overlaid, so many will not match.
- It does not yet recognize every shiny/form/alternate artwork reliably. A later version should add a trained visual model and curated source-specific references.
- It only sees channels where the Discord bot has permission to view the channel and read message history, and only if Message Content Intent is enabled.
- It uses a regular bot account, not a user self-bot.

## Setup on Android / Termux

1. Install Python and Git in Termux:
   ```sh
   pkg update && pkg upgrade -y
   pkg install python git -y
   ```
2. Clone this repository and enter its folder:
   ```sh
   git clone https://github.com/DevRedant/Pokemon-Detector.git
   cd Pokemon-Detector
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Create your environment file:
   ```sh
   cp .env.example .env
   ```
5. Edit `.env` and set `DISCORD_TOKEN` and `OWNER_ID`.
6. In the Discord Developer Portal, enable **Message Content Intent**. Invite the bot with permissions to view channels, read message history, send messages, embed links, and connect/speak if you want voice commands.
7. Start:
   ```sh
   python bot.py
   ```

On first start the bot automatically downloads available Pokémon image references from PokeAPI. This may take a while and uses mobile data. The images/fingerprints are cached in `data/pokemon.db`. To refresh later, use `!dataset sync` (owner only).

## Commands
- `!help`
- `!feedback your message`
- `!dataset`
- `!dataset sync` (owner only)
- `!status online|idle|dnd|invisible` (owner only)
- `!join` and `!leave` (owner only)

When the owner replies directly to the forwarded feedback message in their DM with the bot, the bot relays that reply to the original user by DM.

## Keep the token private
Never commit `.env`, screenshots containing your token, or the token itself. If it leaks, reset it in the Discord Developer Portal immediately.
