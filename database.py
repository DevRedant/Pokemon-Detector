import sqlite3
from config import DATABASE

def connect():
    db = sqlite3.connect(DATABASE, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    return db

def init_db():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pokemon_name TEXT NOT NULL,
                image_url TEXT NOT NULL UNIQUE,
                image_hash TEXT NOT NULL,
                variant TEXT NOT NULL DEFAULT 'normal',
                source TEXT NOT NULL DEFAULT 'PokeAPI',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        columns = {r["name"] for r in db.execute("PRAGMA table_info(pokemon_images)")}
        if "updated_at" not in columns:
            db.execute("ALTER TABLE pokemon_images ADD COLUMN updated_at TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS idx_pokemon_hash ON pokemon_images(image_hash)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                owner_message_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS detected_messages (
                message_id INTEGER PRIMARY KEY,
                detected_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_image(name, url, image_hash, variant="normal", source="PokeAPI"):
    with connect() as db:
        db.execute("""
            INSERT INTO pokemon_images(pokemon_name,image_url,image_hash,variant,source)
            VALUES(?,?,?,?,?)
            ON CONFLICT(image_url) DO UPDATE SET
              pokemon_name=excluded.pokemon_name,
              image_hash=excluded.image_hash,
              variant=excluded.variant,
              source=excluded.source,
              updated_at=CURRENT_TIMESTAMP
        """, (name, url, image_hash, variant, source))

def get_all_hashes():
    with connect() as db:
        return db.execute("SELECT pokemon_name,image_hash,variant FROM pokemon_images").fetchall()

def image_count():
    with connect() as db:
        return db.execute("SELECT COUNT(*) FROM pokemon_images").fetchone()[0]

def save_feedback(owner_message_id, user_id):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO feedback(owner_message_id,user_id) VALUES(?,?)",
                   (owner_message_id, user_id))

def get_feedback_user(owner_message_id):
    with connect() as db:
        row = db.execute("SELECT user_id FROM feedback WHERE owner_message_id=?",
                         (owner_message_id,)).fetchone()
        return row["user_id"] if row else None

def was_detected(message_id):
    with connect() as db:
        return db.execute("SELECT 1 FROM detected_messages WHERE message_id=?",
                          (message_id,)).fetchone() is not None

def mark_detected(message_id, name):
    with connect() as db:
        db.execute("INSERT OR IGNORE INTO detected_messages(message_id,detected_name) VALUES(?,?)",
                   (message_id, name))
