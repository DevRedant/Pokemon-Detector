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


def fingerprint_image_crops(image):
    """Fingerprint the full scene and centered crops to reduce background influence."""
    image = image.convert("RGB")
    width, height = image.size
    fingerprints = [fingerprint_image(image)]

    # Pokémon spawn artwork is usually near the center. A square crop removes
    # much of the surrounding scene while retaining the central subject.
    side = min(width, height)
    center_x, center_y = width // 2, height // 2
    for scale in (1.0, 0.9, 0.8, 0.7):
        crop_side = max(1, int(side * scale))
        left = max(0, center_x - crop_side // 2)
        top = max(0, center_y - crop_side // 2)
        right = min(width, left + crop_side)
        bottom = min(height, top + crop_side)
        crop = image.crop((left, top, right, bottom))
        if crop.size != image.size:
            fingerprints.append(fingerprint_image(crop))

    return fingerprints

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
    if not query_hashes:
        return None
    # Compare each stored reference against every crop and keep its best score.
    ranked = []
    for row in get_all_hashes():
        distances = [
            _distance(query, row["image_hash"])
            for query in query_hashes
        ]
        distances = [distance for distance in distances if distance is not None]
        if distances:
            ranked.append((min(distances), row["pokemon_name"], row["variant"]))
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
            hashes = await asyncio.to_thread(fingerprint_image_crops, image.copy())
        return hashes, raw
    except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
        return None

async def detect_url(url, session):
    image_data = await read_image_url(url, session)
    if image_data is None:
        log.warning("Could not read image for Pokémon detection: %s", url)
        return None
    hashes, raw = image_data

    # Keep the lightweight existing matcher as the first pass.
    match = best_match(hashes)
    if match:
        return (*match, "hash")

    # If no hash match exists, try the optional local ONNX model.
    try:
        from ai_detector import predict_image, model_status
        if model_status()["available"]:
            prediction = await asyncio.to_thread(predict_image, raw)
            if prediction:
                name, confidence = prediction
                log.info("Local AI predicted %s with confidence %.3f", name, confidence)
                return name.replace("_", " ").title(), "normal", round(confidence, 3), "AI"
    except Exception:
        log.exception("Local AI fallback failed; continuing without it.")
    return None
