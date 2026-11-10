import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
PREFIX = os.getenv("PREFIX", "!")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "14"))
MATCH_MIN_MARGIN = float(os.getenv("MATCH_MIN_MARGIN", "1.5"))
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", str(8 * 1024 * 1024)))
DATABASE = os.getenv("DATABASE", "data/pokemon.db")
DATASET_CONCURRENCY = max(1, min(16, int(os.getenv("DATASET_CONCURRENCY", "8"))))
DETECTION_CONCURRENCY = max(1, min(8, int(os.getenv("DETECTION_CONCURRENCY", "3"))))

if not PREFIX or len(PREFIX) > 5:
    raise RuntimeError("PREFIX must contain between 1 and 5 characters.")
if MAX_IMAGE_SIZE < 1024:
    raise RuntimeError("MAX_IMAGE_SIZE must be at least 1024 bytes.")

Path(DATABASE).parent.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing. Set it as an environment variable.")
if OWNER_ID <= 0:
    raise RuntimeError("Set OWNER_ID to your numeric Discord user ID.")
