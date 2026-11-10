import asyncio
import logging
import discord
from discord.ext import commands
import aiohttp

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
intents.members = False
intents.guilds = True
intents.messages = True
intents.voice_states = True

bot = commands.Bot(command_prefix=config.PREFIX, intents=intents,
                   help_command=None, case_insensitive=True)
http_session = None
dataset_task = None

def owner_only():
    async def predicate(ctx):
        return ctx.author.id == config.OWNER_ID
    return commands.check(predicate)

async def get_image_urls(message):
    urls = []
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            urls.append(attachment.url)
        elif attachment.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            urls.append(attachment.url)
    for embed in message.embeds:
        if embed.image and embed.image.url:
            urls.append(embed.image.url)
        if embed.thumbnail and embed.thumbnail.url:
            urls.append(embed.thumbnail.url)
    return list(dict.fromkeys(urls))

async def background_dataset_sync():
    await bot.wait_until_ready()
    if image_count() > 0:
        log.info("Dataset cache already contains %d images; use %sdataset sync to refresh.",
                 image_count(), config.PREFIX)
        return
    try:
        log.info("Starting first automatic PokeAPI dataset download...")
        total = await sync_dataset()
        log.info("Dataset setup finished: %d image fingerprints saved.", total)
    except Exception:
        log.exception("Automatic dataset download failed. Try %sdataset sync later.", config.PREFIX)

@bot.event
async def on_ready():
    global http_session, dataset_task
    if http_session is None or http_session.closed:
        timeout = aiohttp.ClientTimeout(total=25, connect=10)
        http_session = aiohttp.ClientSession(timeout=timeout,
                                             headers={"User-Agent": "PokeDetector/1.0"})
    log.info("Logged in as %s (%s)", bot.user, bot.user.id)
    if dataset_task is None or dataset_task.done():
        dataset_task = asyncio.create_task(background_dataset_sync())

@bot.event
async def on_message(message):
    if message.author.id == bot.user.id:
        return

    if message.author.id == config.OWNER_ID and isinstance(message.channel, discord.DMChannel):
        reference = message.reference
        if reference and reference.message_id:
            user_id = get_feedback_user(reference.message_id)
            if user_id:
                try:
                    user = await bot.fetch_user(user_id)
                    await user.send(f"💬 **Reply from the bot owner:**\n{message.content}")
                    await message.add_reaction("✅")
                except discord.HTTPException:
                    await message.add_reaction("⚠️")

    if message.guild is not None:
        urls = await get_image_urls(message)
        if urls and not was_detected(message.id):
            for url in urls[:4]:
                result = await detect_url(url, http_session)
                if result:
                    name, variant, distance = result
                    mark_detected(message.id, name)
                    label = " ✨ **Shiny**" if variant == "shiny" else ""
                    await message.channel.send(
                        f"🔎 I think this Pokémon is **{name}**{label}. "
                        f"({distance}/64 image distance)",
                        reference=message,
                        mention_author=False
                    )
                    break
    await bot.process_commands(message)

@bot.command(name="help")
async def help_command(ctx):
    await ctx.send(
        f"**Pokémon Detector**\n"
        f"`{config.PREFIX}feedback <message>` — send feedback to the owner\n"
        f"`{config.PREFIX}status <online|idle|dnd|invisible>` — owner only\n"
        f"`{config.PREFIX}join` / `{config.PREFIX}leave` — owner only, voice\n"
        f"`{config.PREFIX}dataset` — show cached reference count\n"
        f"`{config.PREFIX}dataset sync` — owner-only refresh from PokeAPI"
    )

@bot.command()
async def feedback(ctx, *, text: str):
    if not text.strip():
        return await ctx.reply("Please include your feedback.")
    owner = await bot.fetch_user(config.OWNER_ID)
    embed = discord.Embed(title="New Pokémon Detector feedback",
                          description=text[:4000], color=0x55aaff)
    embed.add_field(name="From", value=f"{ctx.author} (`{ctx.author.id}`)", inline=False)
    embed.add_field(name="Location", value=f"{ctx.guild.name if ctx.guild else 'DM'}", inline=False)
    owner_message = await owner.send(embed=embed)
    save_feedback(owner_message.id, ctx.author.id)
    await ctx.reply("✅ Your feedback was sent to the bot owner.")

@bot.group(invoke_without_command=True)
async def dataset(ctx):
    await ctx.send(f"📚 Cached reference images: **{image_count()}**")

@dataset.command(name="sync")
@owner_only()
async def dataset_sync(ctx):
    msg = await ctx.reply("⏳ Updating the automatic Pokémon image dataset. This may take a while...")
    try:
        count = await sync_dataset()
        await msg.edit(content=f"✅ Dataset sync finished. Saved/updated **{count}** image fingerprints. Total cached: **{image_count()}**.")
    except Exception as exc:
        log.exception("Dataset sync failed")
        await msg.edit(content=f"❌ Dataset sync failed: `{str(exc)[:300]}`")

@bot.command()
@owner_only()
async def status(ctx, state: str):
    choices = {
        "online": discord.Status.online,
        "idle": discord.Status.idle,
        "dnd": discord.Status.dnd,
        "invisible": discord.Status.invisible,
    }
    status_value = choices.get(state.lower())
    if status_value is None:
        return await ctx.reply("Use: online, idle, dnd or invisible.")
    await bot.change_presence(status=status_value)
    await ctx.reply(f"✅ Status changed to **{state.lower()}**.")

@bot.command()
@owner_only()
async def join(ctx):
    if not ctx.guild or not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.reply("Join a voice channel first.")
    channel = ctx.author.voice.channel
    voice = ctx.guild.voice_client
    if voice:
        await voice.move_to(channel)
    else:
        await channel.connect()
    await ctx.reply(f"🔊 Joined **{channel.name}**.")

@bot.command()
@owner_only()
async def leave(ctx):
    voice = ctx.guild.voice_client if ctx.guild else None
    if not voice:
        return await ctx.reply("I'm not in a voice channel.")
    await voice.disconnect()
    await ctx.reply("👋 Left the voice channel.")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return await ctx.reply(f"❓ Unknown command. Use `{config.PREFIX}help`.")
    if isinstance(error, commands.CheckFailure):
        return await ctx.reply("⛔ This command is owner-only.")
    if isinstance(error, commands.MissingRequiredArgument):
        return await ctx.reply(f"Missing information. Use `{config.PREFIX}help`.")
    log.error("Command error: %s", error)
    await ctx.reply("Something went wrong while running that command.")

async def main():
    global http_session
    init_db()
    try:
        async with bot:
            await bot.start(config.TOKEN)
    finally:
        if http_session and not http_session.closed:
            await http_session.close()

if __name__ == "__main__":
    asyncio.run(main())
