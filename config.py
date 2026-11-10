import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
PREFIX = os.getenv("PREFIX", "!")
MATCH_THRESHOLD = int(os.getenv("MATCH_THRESHOLD", "8"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", str(8 * 1024 * 1024)))
DATABASE = os.getenv("DATABASE", "data/pokemon.db")
Path(DATABASE).parent.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing. Copy .env.example to .env and add your bot token.")
if OWNER_ID <= 0:
    raise RuntimeError("Set OWNER_ID in .env to your Discord user ID.")
