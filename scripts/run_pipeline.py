"""
run_pipeline.py — run one week end to end, stopping at the first failing stage.

    heroes   ->  ingest  ->  validate  ->  transform  ->  charts         ->  insights
    (sync)       (parquet)   (quality)     (SQL→CSV)      (gen_visuals)      (outline + scenes CSV)

Usage:
    python scripts/run_pipeline.py                  # last completed week
    python scripts/run_pipeline.py --week 12
    python scripts/run_pipeline.py --week 12 --from transform     # skip earlier stages
    python scripts/run_pipeline.py --skip heroes
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pipeline_common import RANKED_START_DATE, parse_date, setup_logging, week_bounds, week_number

HERE = Path(__file__).resolve().parent
log = setup_logging()


def find_script(name: str) -> Path:
    """Chart-side scripts live in whichever scripts/ subfolder holds the chart scripts."""
    hits = sorted(HERE.rglob(name))
    if not hits:
        raise SystemExit(f"{name} not found under scripts/")
    return hits[0]


def stages(week: int) -> list[tuple[str, list[str]]]:
    chart_date = week_bounds(week)[1].date().isoformat()     # gen_visuals picks the week from a date
    w = ["--week", str(week)]
    return [
        ("heroes", [str(HERE / "ingest_scripts" / "hero_ingest_deadlock.py")]),
        ("ingest", [str(HERE / "ingest_scripts" / "ingest_dl_comm_api_parquet.py"), *w]),
        ("validate", [str(HERE / "ingest_scripts" / "validate_week.py"), *w]),
        ("transform", [str(HERE / "sql_scripts" / "query_transform.py"), *w]),
        ("charts", [str(find_script("gen_visuals.py")), "--start-date", chart_date]),
        ("insights", [str(find_script("insights.py")), *w]),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--week", type=int)
    g.add_argument("--date", "--start-date", dest="date")
    ap.add_argument("--from", dest="start_at", choices=["heroes", "ingest", "validate", "transform", "charts", "insights"])
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to skip")
    args = ap.parse_args()

    week = args.week or week_number(parse_date(args.date))
    plan = stages(week)
    if args.start_at:
        plan = plan[[n for n, _ in plan].index(args.start_at):]
    plan = [(n, cmd) for n, cmd in plan if n not in args.skip]
    log.info("Week %s (ranked launch %s): %s", week, RANKED_START_DATE, " -> ".join(n for n, _ in plan))

    for name, cmd in plan:
        t = time.time()
        log.info("=== %s", name)
        if subprocess.run([sys.executable, *cmd]).returncode != 0:
            log.error("Stage '%s' failed after %.0fs — stopping. Fix it, then resume with --from %s",
                      name, time.time() - t, name)
            return 1
        log.info("=== %s done in %.0fs", name, time.time() - t)
    return 0


if __name__ == "__main__":
    sys.exit(main())