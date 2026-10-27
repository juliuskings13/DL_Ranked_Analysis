"""
db_config.py — single source of PostgreSQL connection settings.

Values come from environment variables, loaded from a `.env` file at the repo root
if python-dotenv is installed. Never hard-code credentials in scripts; import:

    from db_config import PG_CONFIG, DB_URI
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # repo-root .env
    load_dotenv()                                                  # or one in the cwd
except ImportError:
    pass  # fine if variables are exported in the shell instead


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. Copy .env.example to .env and fill it in."
        )
    return value


PG_CONFIG = {
    "dbname": os.environ.get("PG_DBNAME", "deadlock_db"),
    "user": _require("PG_USER"),
    "password": _require("PG_PASSWORD"),
    "host": os.environ.get("PG_HOST", "localhost"),
    "port": int(os.environ.get("PG_PORT", "5432")),
}

DB_URI = (
    f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
    f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['dbname']}"
)
