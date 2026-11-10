"""Automatic reference-image collector using the public PokeAPI."""
import asyncio
import io
import logging
import aiohttp
from PIL import Image
import imagehash
from database import save_image

log = logging.getLogger("pokedetector.dataset")
API = "https://pokeapi.co/api/v2"

def image_urls(detail):
    """Collect distinct official artwork and sprite URLs returned by PokeAPI."""
    found = {}
    sprites = detail.get("sprites") or {}

    def add(url, variant):
        if isinstance(url, str) and url.startswith("http"):
            found[url] = variant

    add(sprites.get("front_default"), "normal")
    add(sprites.get("front_shiny"), "shiny")

    data = sprites.get("other") or {}
    for key in ("official-artwork", "home"):
        art = data.get(key) or {}
        add(art.get("front_default"), "normal")
        add(art.get("front_shiny"), "shiny")
    return found

async def fetch_json(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()

async def download_hash(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                raw = await response.content.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    return None
            with Image.open(io.BytesIO(raw)) as img:
                img = img.convert("RGB")
                img.thumbnail((512, 512))
                return str(imagehash.phash(img))
        except Exception as exc:
            log.debug("Image skipped %s: %s", url, exc)
            return None

async def sync_dataset():
    """Fetch Pokémon records and fingerprint their available PokeAPI images."""
    timeout = aiohttp.ClientTimeout(total=35, connect=12)
    headers = {"User-Agent": "PokeDetector/1.0 (personal Discord bot)"}
    connector = aiohttp.TCPConnector(limit=16, ttl_dns_cache=300)
    semaphore = asyncio.Semaphore(8)
    async with aiohttp.ClientSession(timeout=timeout, headers=headers,
                                     connector=connector) as session:
        listing = await fetch_json(session, f"{API}/pokemon?limit=2000")
        entries = listing.get("results", [])
        log.info("PokeAPI listed %d Pokémon/form records.", len(entries))

        async def process(entry):
            try:
                detail = await fetch_json(session, entry["url"])
                name = detail.get("name", entry["name"]).replace("-", " ").title()
                for url, variant in image_urls(detail).items():
                    fingerprint = await download_hash(session, url, semaphore)
                    if fingerprint:
                        save_image(name, url, fingerprint, variant)
                        yield_count[0] += 1
            except Exception as exc:
                log.warning("Could not process %s: %s", entry.get("name"), exc)

        yield_count = [0]
        queue = asyncio.Semaphore(10)
        async def bounded(entry):
            async with queue:
                await process(entry)
        await asyncio.gather(*(bounded(entry) for entry in entries))
        return yield_count[0]
