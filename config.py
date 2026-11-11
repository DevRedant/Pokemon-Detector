from pathlib import Path
TOKEN = "PASTE_YOUR_BOT_TOKEN_HERE"
OWNER_ID = 0  # Replace with your numeric Discord user ID
PREFIX = "."
MATCH_THRESHOLD = 14.0
MATCH_MIN_MARGIN = 1.5
MAX_IMAGE_SIZE = 8 * 1024 * 1024
DATABASE = "data/pokemon.db"
DATASET_CONCURRENCY = 8
DETECTION_CONCURRENCY = 3

if not PREFIX or len(PREFIX) > 5:
    raise RuntimeError("PREFIX must contain between 1 and 5 characters.")
if MAX_IMAGE_SIZE < 1024:
    raise RuntimeError("MAX_IMAGE_SIZE must be at least 1024 bytes.")

Path(DATABASE).parent.mkdir(parents=True, exist_ok=True)

if TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
    raise RuntimeError("Paste your Discord bot token into TOKEN in config.py.")
if OWNER_ID <= 0:
    raise RuntimeError("Set OWNER_ID to your numeric Discord user ID in config.py.")
