"""SQLite storage for results and the human-review queue."""
import sqlite3, json
from config import DB_PATH

SCHEMA = """CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, category TEXT, sentiment TEXT, priority TEXT,
  status TEXT, confidence REAL, llm_calls INTEGER, latency REAL, trace TEXT,
  reviewed INTEGER DEFAULT 0, created TEXT DEFAULT CURRENT_TIMESTAMP)"""


def conn():
    c = sqlite3.connect(DB_PATH)
    c.execute(SCHEMA)
    return c


def save(r: dict) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO results (text,category,sentiment,priority,status,confidence,llm_calls,latency,trace) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (r["text"], r["category"], r["sentiment"], r["priority"], r["status"],
             r["confidence"], r["llm_calls"], r["latency"], json.dumps(r["trace"])))
        return cur.lastrowid


def correct(row_id: int, category: str, sentiment: str):
    with conn() as c:
        c.execute("UPDATE results SET category=?, sentiment=?, status='human_corrected', reviewed=1 WHERE id=?",
                  (category, sentiment, row_id))
