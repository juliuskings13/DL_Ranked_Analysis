"""
ingest_dl_comm_api_database.py — small dimension tables via the Deadlock API's SQL endpoint.

Currently syncs `items` (for future item/build analysis). Match data is NOT loaded here any
more: the parquet ingest (ingest_dl_comm_api_parquet.py) is the single source for
match_player, because the two scripts used to create match_player with different columns
and whichever ran first broke the other.

The API client below is reusable for any other small table: bounded retries with
exponential backoff on 429 / 5xx, a request timeout, and keyset pagination
(WHERE key > last_key) instead of LIMIT/OFFSET, which gets slower every page and can skip
or repeat rows if the table changes between requests.

Usage: python ingest_dl_comm_api_database.py
"""
from __future__ import annotations

import sys
import time

import requests

from pipeline_common import connect, copy_upsert, ensure_schema, setup_logging

SQL_URL = "https://api.deadlock-api.com/v1/sql"
HEADERS = {"User-Agent": "Deadlock Analytics (personal project)"}
log = setup_logging()


def query_api(sql: str, max_attempts: int = 6) -> list[dict]:
    """Run a ClickHouse query on the community API; retries 429/5xx with backoff (5s→120s)."""
    for attempt in range(1, max_attempts + 1):
        try:
            res = requests.get(SQL_URL, params={"query": sql, "format": "json"}, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            status, detail = None, str(e)
        else:
            if res.status_code == 200:
                return res.json()
            status, detail = res.status_code, res.text[:200]
            if status not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"API error {status}: {detail}")
        if attempt == max_attempts:
            raise RuntimeError(f"API still failing after {max_attempts} attempts ({status}: {detail})")
        wait = min(5 * 2 ** (attempt - 1), 120)
        log.warning("API %s, retrying in %ss (attempt %s/%s)", status or "connection error", wait, attempt,
                    max_attempts)
        time.sleep(wait)
    return []


def query_api_paged(select: str, table: str, key: str, page: int = 5000, where: str = "1"):
    """Yield all rows of a query page by page using keyset pagination on a unique numeric key."""
    last = -1
    while True:
        rows = query_api(f"SELECT {select} FROM {table} WHERE ({where}) AND {key} > {last} "
                         f"ORDER BY {key} LIMIT {page}")
        if not rows:
            return
        yield rows
        last = rows[-1][key]
        time.sleep(2)                      # be polite to a free community API


def sync_items(conn) -> int:
    cols = ["id", "name", "tier", "type", "slot_type", "cost"]
    total = 0
    for rows in query_api_paged(", ".join(cols), "items", key="id"):
        total += copy_upsert(conn, "items", cols, [tuple(r.get(c) for c in cols) for r in rows], key=["id"],
                             conflict="DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[1:]))
    return total


def main() -> int:
    conn = connect()
    try:
        ensure_schema(conn)
        log.info("Synced %s items", sync_items(conn))
        return 0
    except Exception:
        log.exception("Dimension sync failed")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())