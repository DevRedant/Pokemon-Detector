import asyncio
import io
import logging
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import aiohttp
import discord
from discord.ext import commands

import config
from database import (
    init_db, image_count, save_feedback, get_feedback_user,
    was_detected, mark_detected
)
from dataset import sync_dataset
from detector import detect_url

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("pokedetector")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.voice_states = True

bot = commands.Bot(command_prefix=config.PREFIX, intents=intents,
                   help_command=None, case_insensitive=True)
http_session = None
dataset_task = None
dataset_lock = asyncio.Lock()
detection_semaphore = asyncio.Semaphore(config.DETECTION_CONCURRENCY)
pending_messages = set()

def make_embed(title, description=None, color=discord.Color.blurple()):
    embed = discord.Embed(title=title, description=description, color=color)
    embed.set_footer(text="Pokémon Detector")
    return embed

async def reply_embed(ctx, title, description=None, *, color=discord.Color.blurple()):
    return await ctx.reply(embed=make_embed(title, description, color=color))
def owner_only():
    async def predicate(ctx):
        return ctx.author.id == config.OWNER_ID
    return commands.check(predicate)

async def get_image_urls(message):
    urls = []
    for attachment in message.attachments:
        content_type = (attachment.content_type or "").lower()
        filename = attachment.filename.lower()
        if content_type.startswith("image/") or filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif")):
            urls.append(attachment.url)
    for embed in message.embeds:
        for item in (embed.image, embed.thumbnail):
            if item and item.url:
                urls.append(item.url)
    return list(dict.fromkeys(urls))

async def background_dataset_sync():
    await bot.wait_until_ready()
    if image_count() > 0:
        log.info("Dataset cache contains %d images.", image_count())
        return
    if dataset_lock.locked():
        log.info("A dataset sync is already running; skipping automatic sync.")
        return
    async with dataset_lock:
        if image_count() > 0:
            return
        try:
            log.info("Starting initial PokeAPI dataset download.")
            count = await sync_dataset()
            log.info("Initial dataset sync saved %d images.", count)
        except Exception:
            log.exception("Initial dataset sync failed; use %sdataset sync to retry.", config.PREFIX)

async def process_images(message):
    try:
        if http_session is None or http_session.closed or was_detected(message.id):
            return
        async with detection_semaphore:
            image_urls = (await get_image_urls(message))[:4]
            log.info(
                "Checking message %s from %s (%s): %d image URL(s)",
                message.id, message.author, "bot" if message.author.bot else "user",
                len(image_urls)
            )
            for url in image_urls:
                log.info("Downloading image for message %s: %s", message.id, url)
                result = await detect_url(url, http_session)
                if result:
                    name, variant, score, method = result
                    mark_detected(message.id, name)
                    shiny = " ✨ Shiny" if variant == "shiny" else ""
                    detail = f"AI confidence: {score:.1%}" if method == "AI" else f"Hash distance: {score}/64"
                    result_embed = make_embed("Pokémon detected!",
                        f"## {name}{shiny}\n**Detection method:** {method}\n**Match detail:** {detail}",
                        color=discord.Color.green())
                    await message.channel.send(embed=result_embed, reference=message,
                        mention_author=False, allowed_mentions=discord.AllowedMentions.none())
                    log.info("Detected %s (variant=%s, method=%s, score=%s) in message %s",
                             name, variant, method, score, message.id)
                    return
                log.info("No Pokémon match for image in message %s", message.id)
    except (discord.HTTPException, aiohttp.ClientError):
        log.exception("Could not process image message %s", message.id)
    except Exception:
        log.exception("Unexpected image-processing error for message %s", message.id)
    finally:
        pending_messages.discard(message.id)

@bot.event
async def on_ready():
    global http_session, dataset_task
    if http_session is None or http_session.closed:
        timeout = aiohttp.ClientTimeout(total=25, connect=10)
        http_session = aiohttp.ClientSession(timeout=timeout,
                                             headers={"User-Agent": "PokemonDetectorDiscordBot/2.0"})
    log.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")
    if dataset_task is None or dataset_task.done():
        dataset_task = asyncio.create_task(background_dataset_sync())

@bot.event
async def on_message(message):
    if message.guild and message.author.bot and (message.embeds or message.attachments):
        log.info(
            "Received bot image candidate: author=%s (%s), embeds=%d, attachments=%d",
            message.author, message.author.id, len(message.embeds), len(message.attachments)
        )

    if bot.user and message.author.id == bot.user.id:
        return

    # Owner replies to the forwarded feedback message in the bot DM.
    if message.author.id == config.OWNER_ID and isinstance(message.channel, discord.DMChannel):
        ref = message.reference
        if ref and ref.message_id:
            user_id = get_feedback_user(ref.message_id)
            if user_id:
                try:
                    user = await bot.fetch_user(user_id)
                    await user.send(f"💬 **Reply from the bot owner:**\n{message.content[:1900]}")
                    await message.add_reaction("✅")
                except discord.HTTPException:
                    await message.add_reaction("⚠️")

    # Run commands promptly; image downloads happen in a separate bounded task.
    await bot.process_commands(message)
    if message.guild is not None:
        if message.id not in pending_messages:
            urls = await get_image_urls(message)
            if urls and not was_detected(message.id):
                pending_messages.add(message.id)
                asyncio.create_task(process_images(message))

@bot.command(name="ai")
@owner_only()
async def ai_status(ctx):
    from ai_detector import model_status
    status = model_status()
    root = Path("data/training")
    classes = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    samples = sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    state = "✅ Model files found" if status["available"] else "⚠️ Model not trained/exported yet"
    embed = make_embed("Local AI status", state,
        color=discord.Color.green() if status["available"] else discord.Color.orange())
    embed.add_field(name="Training examples", value=f"{samples} across {len(classes)} Pokémon labels", inline=True)
    embed.add_field(name="Model", value=f"`{status['model']}`", inline=False)
    embed.add_field(name="Labels", value=f"`{status['labels']}`", inline=False)
    embed.add_field(name="Confidence threshold", value=__import__('os').getenv('AI_CONFIDENCE', '0.65'), inline=True)
    await ctx.reply(embed=embed)

@bot.command(name="learn")
@owner_only()
async def learn_spawn(ctx, *, pokemon: str):
    """Save a correctly labelled spawn image by replying to that image."""
    if http_session is None or http_session.closed:
        return await reply_embed(ctx, "Image downloader is not ready yet", "Image downloader is not ready yet.", color=discord.Color.orange())
    reference = ctx.message.reference
    if not reference or not reference.message_id:
        return await reply_embed(ctx, "Reply directly to a Pokémon spawn image with `", "Reply directly to a Pokémon spawn image with `.learn Pikachu`.", color=discord.Color.orange())
    try:
        target = reference.resolved
        if target is None:
            target = await ctx.channel.fetch_message(reference.message_id)
    except discord.HTTPException:
        return await reply_embed(ctx, "I couldn't read the message you replied to", "I couldn't read the message you replied to.", color=discord.Color.orange())
    label = re.sub(r"[^a-z0-9_-]+", "_", pokemon.strip().lower()).strip("_")
    if not label:
        return await reply_embed(ctx, "Please provide a valid Pokémon name", "Please provide a valid Pokémon name.", color=discord.Color.orange())
    urls = (await get_image_urls(target))[:4]
    if not urls:
        return await reply_embed(ctx, "The replied message has no accessible image", "The replied message has no accessible image.", color=discord.Color.orange())
    folder = Path("data/training") / label
    folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    for index, url in enumerate(urls, start=1):
        try:
            async with http_session.get(url, allow_redirects=True) as response:
                if response.status != 200 or not response.headers.get("Content-Type", "").lower().startswith("image/"):
                    continue
                raw = await response.content.read(config.MAX_IMAGE_SIZE + 1)
                if len(raw) > config.MAX_IMAGE_SIZE:
                    continue
            def write_image(data, destination):
                with Image.open(io.BytesIO(data)) as source:
                    image = source.convert("RGB")
                    image.thumbnail((1600, 1600))
                    image.save(destination, format="JPEG", quality=92, optimize=True)
            destination = folder / f"{target.id}_{index}.jpg"
            await asyncio.to_thread(write_image, raw, destination)
            saved += 1
        except (aiohttp.ClientError, OSError, UnidentifiedImageError, ValueError):
            log.exception("Could not save labelled training image from %s", url)
    if saved:
        await ctx.reply(embed=make_embed("Training image saved",
            f"Saved **{saved}** image(s) as **{label.replace('_', ' ').title()}**.\nCollect more varied examples before training.",
            color=discord.Color.green()))
    else:
        await reply_embed(ctx, "❌ I couldn't save that image", "❌ I couldn't save that image. Check the bot logs.", color=discord.Color.orange())

@bot.command(name="help")
async def help_command(ctx):
    embed = make_embed("Pokémon Detector • Command Guide", "Commands and tools available in this server.")
    embed.add_field(name="🔎 Detection", value="Automatic Pokémon image detection runs in accessible channels.", inline=False)
    embed.add_field(name="📊 Presence", value=f"`{config.PREFIX}status <online|idle|dnd|invisible> <activity text>` — change status and Playing text (owner only).", inline=False)
    embed.add_field(name="🤖 AI", value=f"`{config.PREFIX}ai` — local AI status (owner only).\n`{config.PREFIX}learn <Pokemon>` — save a labelled image example (owner only).", inline=False)
    embed.add_field(name="📚 Dataset", value=f"`{config.PREFIX}dataset` — cached image count.\n`{config.PREFIX}dataset sync` — refresh image references (owner only).", inline=False)
    embed.add_field(name="🔊 Voice", value=f"`{config.PREFIX}join` / `{config.PREFIX}leave` — voice controls (owner only).", inline=False)
    embed.add_field(name="💬 Feedback", value=f"`{config.PREFIX}feedback <message>` — send feedback to the owner.", inline=False)
    await ctx.send(embed=embed)

@bot.command()
async def feedback(ctx, *, text: str):
    text = text.strip()
    if not text:
        return await reply_embed(ctx, "Please include your feedback", "Please include your feedback.", color=discord.Color.orange())
    owner = await bot.fetch_user(config.OWNER_ID)
    embed = discord.Embed(title="New Pokémon Detector feedback",
                          description=text[:4000], color=0x55AAFF)
    embed.add_field(name="From", value=f"{ctx.author} (`{ctx.author.id}`)", inline=False)
    embed.add_field(name="Location", value=ctx.guild.name if ctx.guild else "Direct message", inline=False)
    owner_message = await owner.send(embed=embed)
    save_feedback(owner_message.id, ctx.author.id)
    await reply_embed(ctx, "✅ Your feedback was sent to the bot owner", "✅ Your feedback was sent to the bot owner.", color=discord.Color.orange())

@bot.group(invoke_without_command=True)
async def dataset(ctx):
    await ctx.send(embed=make_embed("Pokémon Dataset", f"Cached reference images: **{image_count():,}**"))

@dataset.command(name="sync")
@owner_only()
async def dataset_sync(ctx):
    if dataset_lock.locked():
        return await reply_embed(ctx, "⏳ A dataset sync is already running", "⏳ A dataset sync is already running.", color=discord.Color.orange())
    async with dataset_lock:
        msg = await reply_embed(ctx, "⏳ Updating Pokémon reference images from PokeAPI", "⏳ Updating Pokémon reference images from PokeAPI. This may take a while...", color=discord.Color.orange())
        try:
            count = await sync_dataset()
            done = make_embed("Dataset sync complete",
                f"Processed **{count:,}** images.\nTotal cached: **{image_count():,}**.",
                color=discord.Color.green())
            await msg.edit(content=None, embed=done)
        except Exception as exc:
            log.exception("Dataset sync failed")
            await msg.edit(content=None, embed=make_embed("Dataset sync failed",
                f"`{str(exc)[:250]}`", color=discord.Color.red()))

@bot.command()
@owner_only()
async def status(ctx, state: str, *, activity_text: str = None):
    choices = {
        "online": discord.Status.online,
        "idle": discord.Status.idle,
        "dnd": discord.Status.dnd,
        "invisible": discord.Status.invisible,
    }
    value = choices.get(state.lower())
    if value is None:
        return await ctx.reply(
            "Use: `.status online Playing Pokémon`, "
            "`.status idle Watching spawns`, "
            "`.status dnd Detecting Pokémon` or "
            "`.status invisible Taking a break`."
        )

    activity = discord.Game(name=activity_text[:128]) if activity_text else None
    await bot.change_presence(status=value, activity=activity)
    if activity_text:
        await ctx.reply(embed=make_embed("Presence updated",
            f"**Status:** {state.lower()}\n**Activity:** Playing {activity_text[:128]}",
            color=discord.Color.green()))
    else:
        await reply_embed(ctx, "Presence updated", f"**Status:** {state.lower()}\n**Activity:** Cleared", color=discord.Color.green())

@bot.command()
@owner_only()
async def join(ctx):
    if not ctx.guild or not getattr(ctx.author, "voice", None) or not ctx.author.voice.channel:
        return await reply_embed(ctx, "Join a voice channel first", "Join a voice channel first.", color=discord.Color.orange())
    channel = ctx.author.voice.channel
    voice = ctx.guild.voice_client
    if voice:
        await voice.move_to(channel)
    else:
        await channel.connect()
    await reply_embed(ctx, "Voice connected", f"Joined **{channel.name}**.", color=discord.Color.green())

@bot.command()
@owner_only()
async def leave(ctx):
    voice = ctx.guild.voice_client if ctx.guild else None
    if not voice:
        return await reply_embed(ctx, "I'm not in a voice channel", "I'm not in a voice channel.", color=discord.Color.orange())
    await voice.disconnect()
    await reply_embed(ctx, "👋 Left the voice channel", "👋 Left the voice channel.", color=discord.Color.orange())

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return await reply_embed(ctx, "Unknown command", f"That command does not exist. Use `{config.PREFIX}help`.", color=discord.Color.orange())
    if isinstance(error, commands.CheckFailure):
        return await reply_embed(ctx, "⛔ This command is owner-only", "⛔ This command is owner-only.", color=discord.Color.orange())
    if isinstance(error, commands.MissingRequiredArgument):
        return await reply_embed(ctx, "Missing information", f"Use `{config.PREFIX}help` to see command usage.", color=discord.Color.orange())
    if isinstance(error, commands.CommandOnCooldown):
        return await reply_embed(ctx, "⏳ Please wait before using that command again", "⏳ Please wait before using that command again.", color=discord.Color.orange())
    log.error("Command error: %s", error, exc_info=error)
    await reply_embed(ctx, "Something went wrong while running that command", "Something went wrong while running that command.", color=discord.Color.orange())

async def main():
    init_db()
    try:
        async with bot:
            await bot.start(config.TOKEN)
    finally:
        if http_session and not http_session.closed:
            await http_session.close()
        if dataset_task and not dataset_task.done():
            dataset_task.cancel()
            await asyncio.gather(dataset_task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
