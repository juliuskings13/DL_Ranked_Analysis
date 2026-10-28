"""
db_config.py — single source of PostgreSQL connection settings.

Lives in the project root next to `.env`. Values come from environment variables,
loaded from that `.env` if python-dotenv is installed. Never hard-code credentials; import:

    from db_config import PG_CONFIG, DB_URI

Importable from any script in the project after a one-time `pip install -e .`
(see pyproject.toml), regardless of which folder the script is in.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")   # .env next to this file (project root)
except ImportError:
    pass  # fine if variables are exported in the shell instead


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. Copy .env.example to .env and fill it in."
        )
    return value


# Every connection runs in this session time zone, so timestamps without an explicit offset
# (e.g. week boundaries computed in Python) mean the same thing on every machine.
PG_TIMEZONE = os.environ.get("PG_TIMEZONE", "UTC")

PG_CONFIG = {
    "dbname": os.environ.get("PG_DBNAME", "deadlock_db"),
    "user": _require("PG_USER"),
    "password": _require("PG_PASSWORD"),
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", "5432")),
    "options": f"-c timezone={PG_TIMEZONE}",
}

# quote_plus: a password containing @ / : # etc. would otherwise break the URL
DB_URI = (
    f"postgresql://{quote_plus(PG_CONFIG['user'])}:{quote_plus(PG_CONFIG['password'])}"
    f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
    f"?options={quote_plus(PG_CONFIG['options'])}"
)