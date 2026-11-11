import asyncio
import io
import ipaddress
import logging
from urllib.parse import urlparse

import aiohttp
from PIL import Image, UnidentifiedImageError
import imagehash

from config import MATCH_THRESHOLD, MATCH_MIN_MARGIN, MAX_IMAGE_SIZE
from database import get_all_hashes

log = logging.getLogger("pokedetector.detector")

def _safe_image_url(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        host = parsed.hostname.lower()
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False
        try:
            if ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback:
                return False
        except ValueError:
            pass
        return True
    except (TypeError, ValueError):
        return False

def fingerprint_image(image):
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    return (imagehash.phash(image), imagehash.dhash(image), imagehash.whash(image))

def encode_fingerprint(hashes):
    return "v2|" + "|".join(str(item) for item in hashes)

def _distance(query, stored):
    try:
        if stored.startswith("v2|"):
            parts = stored.split("|")
            if len(parts) != 4:
                return None
            refs = [imagehash.hex_to_hash(x) for x in parts[1:]]
            return (2 * (query[0] - refs[0]) + (query[1] - refs[1]) + (query[2] - refs[2])) / 4
        # Backward compatibility with older pHash-only database entries.
        return float(query[0] - imagehash.hex_to_hash(stored))
    except (ValueError, TypeError):
        return None

def best_match(query_hashes):
    if query_hashes is None:
        return None
    ranked = []
    for row in get_all_hashes():
        distance = _distance(query_hashes, row["image_hash"])
        if distance is not None:
            ranked.append((distance, row["pokemon_name"], row["variant"]))
    if not ranked:
        log.warning("No usable reference hashes are available for comparison.")
        return None
    ranked.sort(key=lambda item: item[0])
    best = ranked[0]
    log.info(
        "Top Pokémon candidates: %s (threshold=%s, min_margin=%s)",
        "; ".join(
            f"{name}/{variant}: {distance:.1f}"
            for distance, name, variant in ranked[:5]
        ),
        MATCH_THRESHOLD, MATCH_MIN_MARGIN
    )
    # Ambiguity is measured against a different Pokémon label, not another sprite of the same one.
    second = next((item for item in ranked[1:] if item[1].casefold() != best[1].casefold()), None)
    if best[0] > MATCH_THRESHOLD:
        return None
    if second and (second[0] - best[0]) < MATCH_MIN_MARGIN:
        return None
    return best[1], best[2], round(best[0], 1)

async def read_image_url(url, session):
    if not _safe_image_url(url):
        return None
    try:
        async with session.get(url, allow_redirects=True) as response:
            if response.status != 200:
                return None
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                return None
            raw = await response.content.read(MAX_IMAGE_SIZE + 1)
            if len(raw) > MAX_IMAGE_SIZE:
                return None
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return await asyncio.to_thread(fingerprint_image, image.copy())
    except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
        return None

async def detect_url(url, session):
    hashes = await read_image_url(url, session)
    if hashes is None:
        log.warning("Could not read image for Pokémon detection: %s", url)
        return None
    return best_match(hashes)
