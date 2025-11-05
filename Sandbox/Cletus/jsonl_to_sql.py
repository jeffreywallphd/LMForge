import sqlite3
import json

INPUT_FILE = "labeled_chunks_text_books.jsonl"
DB_FILE = "chunks.db"

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Create table
cur.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    label INTEGER
)
""")

# Load JSONL
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cur.execute("INSERT INTO chunks (text, label) VALUES (?, ?)", (obj["text"], obj.get("label")))

conn.commit()
conn.close()

print("Database initialized with data from JSONL.")

