import asyncio
import io
import logging
from pathlib import Path

import aiohttp
import discord
from discord.ext import commands
from PIL import Image, UnidentifiedImageError
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pokemon-detector")

MODEL_ID = "imzynoxprince/pokemons-image-classifier-gen1-gen9"
MODEL_DIR = Path("model_cache")
CONFIDENCE_THRESHOLD = 0.45
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MODEL = None
PROCESSOR = None
MODEL_LOCK = asyncio.Lock()
HTTP = None

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
bot = commands.Bot(command_prefix=config.PREFIX, intents=intents, help_command=None, case_insensitive=True)

def embed(title, description="", color=discord.Color.blurple()):
    e = discord.Embed(title=title, description=description, color=color)
    e.set_footer(text="Pokémon Detector • Local AI")
    return e

async def load_model():
    global MODEL, PROCESSOR
    if MODEL is not None:
        return
    async with MODEL_LOCK:
        if MODEL is not None:
            return
        log.info("Downloading/loading Pokémon AI model: %s", MODEL_ID)
        def load():
            processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(MODEL_DIR))
            model = AutoModelForImageClassification.from_pretrained(MODEL_ID, cache_dir=str(MODEL_DIR))
            model.eval()
            return processor, model
        PROCESSOR, MODEL = await asyncio.to_thread(load)
        log.info("Pokémon AI model ready (%s classes)", MODEL.config.num_labels)

def image_urls(message):
    urls = []
    for a in message.attachments:
        if (a.content_type or "").startswith("image/") or a.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            urls.append(a.url)
    for e in message.embeds:
        if e.image and e.image.url:
            urls.append(e.image.url)
        elif e.thumbnail and e.thumbnail.url:
            urls.append(e.thumbnail.url)
    return list(dict.fromkeys(urls))[:3]

async def classify(raw):
    await load_model()
    def run():
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            inputs = PROCESSOR(images=im, return_tensors="pt")
        with torch.inference_mode():
            logits = MODEL(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1)
            score, idx = torch.max(probs, dim=-1)
        label = MODEL.config.id2label[int(idx)]
        return label.replace("_", " ").replace("-", " ").title(), float(score)
    return await asyncio.to_thread(run)

@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id)
    await bot.change_presence(activity=discord.Game(name="Detecting Pokémon"))

@bot.event
async def on_message(message):
    await bot.process_commands(message)
    if message.author.id == bot.user.id:
        return
    urls = image_urls(message)
    if not urls or HTTP is None:
        return
    for url in urls:
        try:
            async with HTTP.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200 or not (r.headers.get("Content-Type", "").startswith("image/")):
                    continue
                raw = await r.content.read(MAX_IMAGE_BYTES + 1)
            if len(raw) > MAX_IMAGE_BYTES:
                continue
            name, confidence = await classify(raw)
            if confidence >= CONFIDENCE_THRESHOLD:
                result = embed("Pokémon detected!", f"## {name}\n**AI confidence:** {confidence:.1%}", discord.Color.green())
                await message.channel.send(embed=result, reference=message, mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none())
                return
        except (aiohttp.ClientError, UnidentifiedImageError, OSError, ValueError) as exc:
            log.warning("Could not inspect image: %s", exc)
        except Exception:
            log.exception("AI image detection failed")
            return

@bot.command(name="help")
async def help_command(ctx):
    e = embed("Pokémon Detector — Commands")
    e.add_field(name="Automatic detection", value="I inspect image attachments and embeds, including Pokétwo spawn images, and identify Pokémon using a local AI model.", inline=False)
    e.add_field(name=".status", value="Show the bot's current online status.", inline=False)
    e.add_field(name=".help", value="Show this command guide.", inline=False)
    e.add_field(name="AI model", value="Automatically downloaded on first startup. No manual Pokémon images, labels, or training are required.", inline=False)
    await ctx.send(embed=e)

@bot.command()
async def status(ctx):
    state = "Loaded and ready" if MODEL is not None else "Will load automatically when the first image arrives"
    await ctx.send(embed=embed("Bot status", f"**Bot:** Online\n**AI model:** {state}"))

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(embed=embed("Unknown command", f"Use `{config.PREFIX}help` to see commands.", discord.Color.orange()))
    elif isinstance(error, commands.CommandError):
        log.exception("Command failed", exc_info=error)
        await ctx.send(embed=embed("Command error", "Something went wrong. Check the hosting console.", discord.Color.red()))

async def main():
    global HTTP
    async with aiohttp.ClientSession() as session:
        HTTP = session
        async with bot:
            await bot.start(config.TOKEN)

if __name__ == "__main__":
    torch.set_num_threads(1)
    asyncio.run(main())
