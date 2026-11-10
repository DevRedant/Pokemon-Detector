import io
import aiohttp
from PIL import Image, UnidentifiedImageError
import imagehash
from config import MATCH_THRESHOLD, MAX_IMAGE_SIZE
from database import get_all_hashes

async def read_image_url(url, session):
    try:
        async with session.get(url, allow_redirects=True) as response:
            if response.status != 200:
                return None
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not content_type.startswith("image/"):
                return None
            raw = await response.content.read(MAX_IMAGE_SIZE + 1)
            if len(raw) > MAX_IMAGE_SIZE:
                return None
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB")
            image.thumbnail((512, 512))
            return imagehash.phash(image)
    except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
        return None

def best_match(query_hash):
    if query_hash is None:
        return None
    best = None
    best_distance = 65
    for row in get_all_hashes():
        try:
            distance = query_hash - imagehash.hex_to_hash(row["image_hash"])
        except (ValueError, TypeError):
            continue
        if distance < best_distance:
            best_distance = distance
            best = (row["pokemon_name"], row["variant"], distance)
    if best and best_distance <= MATCH_THRESHOLD:
        return best
    return None

async def detect_url(url, session):
    fingerprint = await read_image_url(url, session)
    return best_match(fingerprint)
