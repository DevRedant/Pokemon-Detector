"""Bounded PokeAPI image collector. No manual reference images are required."""
import asyncio
import io
import logging

import aiohttp
from PIL import Image, UnidentifiedImageError

from config import MAX_IMAGE_SIZE, DATASET_CONCURRENCY
from database import save_image
from detector import fingerprint_image, encode_fingerprint

log = logging.getLogger("pokedetector.dataset")
API = "https://pokeapi.co/api/v2"

def image_urls(detail):
    found = {}
    sprites = detail.get("sprites") or {}

    def add(url, variant):
        if isinstance(url, str) and url.startswith("https://"):
            found[url] = variant

    add(sprites.get("front_default"), "normal")
    add(sprites.get("front_shiny"), "shiny")
    for key in ("official-artwork", "home", "showdown"):
        art = (sprites.get("other") or {}).get(key) or {}
        add(art.get("front_default"), "normal")
        add(art.get("front_shiny"), "shiny")
    return found

async def fetch_json(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

async def download_fingerprint(session, url):
    try:
        async with session.get(url, allow_redirects=False) as response:
            if response.status != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                return None
            raw = await response.content.read(MAX_IMAGE_SIZE + 1)
            if len(raw) > MAX_IMAGE_SIZE:
                return None
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return encode_fingerprint(fingerprint_image(image))
    except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
        return None

async def sync_dataset():
    timeout = aiohttp.ClientTimeout(total=35, connect=12)
    headers = {"User-Agent": "PokemonDetectorDiscordBot/2.0"}
    connector = aiohttp.TCPConnector(limit=DATASET_CONCURRENCY * 2, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(DATASET_CONCURRENCY)
    saved = 0
    async with aiohttp.ClientSession(timeout=timeout, headers=headers, connector=connector) as session:
        listing = await fetch_json(session, f"{API}/pokemon?limit=2000")
        entries = listing.get("results", [])
        log.info("PokeAPI returned %d Pokémon/form records.", len(entries))

        async def process(entry):
            nonlocal saved
            async with semaphore:
                try:
                    detail = await fetch_json(session, entry["url"])
                    name = detail.get("name", entry.get("name", "unknown")).replace("-", " ").title()
                    for url, variant in image_urls(detail).items():
                        fingerprint = await download_fingerprint(session, url)
                        if fingerprint:
                            await asyncio.to_thread(save_image, name, url, fingerprint, variant, "PokeAPI")
                            saved += 1
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as exc:
                    log.warning("Skipped dataset record %s: %s", entry.get("name"), exc)

        # gather is bounded by the semaphore; this avoids unbounded HTTP concurrency.
        await asyncio.gather(*(process(entry) for entry in entries))
    return saved
