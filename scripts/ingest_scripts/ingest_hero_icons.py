"""
ingest_hero_icons.py — download hero image assets from the Deadlock API.

Mirrors the API's folder structure under assets/icons/ (e.g. assets/icons/heroes/...).
Files already on disk are skipped, so re-runs only fetch new heroes / new art. Downloads
run in parallel. The chart icon folders (base_icons / critical_icons / gloat_icons) are
separate and not touched here.

Usage: python ingest_hero_icons.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from pipeline_common import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]       # scripts/ingest_scripts/ -> project root
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
ICONS_URL = "https://api.deadlock-api.com/v1/assets/icons"
HEADERS = {"User-Agent": "Deadlock Analytics (personal project)"}
log = setup_logging()


def flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    """{'heroes': {'abrams': {'card': url}}} -> {'heroes/abrams/card': url}"""
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = v
    return out


def download(session: requests.Session, url: str, dest: Path) -> None:
    res = session.get(url, timeout=30)
    res.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(res.content)
    tmp.replace(dest)                     # atomic: a crash never leaves a half-written image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    res = requests.get(ICONS_URL, headers=HEADERS, timeout=30)
    res.raise_for_status()
    assets = flatten(res.json().get("heroes", {}), "heroes/")
    jobs = {path: url for path, url in assets.items()
            if isinstance(url, str) and url.endswith((".png", ".svg"))}
    # the API key keeps its extension in the URL, not always in the path
    todo = [(url, ICONS_DIR / (path if Path(path).suffix else path + Path(url).suffix))
            for path, url in jobs.items()]
    if not args.force:
        todo = [(u, d) for u, d in todo if not d.exists()]
    log.info("%s hero assets listed, %s to download", len(jobs), len(todo))

    failed = 0
    with requests.Session() as s, ThreadPoolExecutor(max_workers=8) as pool:
        s.headers.update(HEADERS)
        futures = {pool.submit(download, s, u, d): d for u, d in todo}
        for f in as_completed(futures):
            try:
                f.result()
            except requests.RequestException as e:
                failed += 1
                log.warning("failed %s: %s", futures[f].relative_to(ICONS_DIR), e)
    log.info("Downloaded %s, failed %s -> %s", len(todo) - failed, failed, ICONS_DIR)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())