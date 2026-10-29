"""
animate.py — turn the weekly chart cards into 1080x1080 video clips.

Every chart script exposes cards(data_dir) -> {name: draw(progress, highlight)}. This script
renders each card's build-in (progress 0 -> 1) frame by frame, holds the finished card, then
plays any highlight "beats" (dim everything except the named heroes / archetypes), and pipes
the frames straight into ffmpeg. Only the build-in frames are actually drawn; holds repeat the
last frame and highlight changes are cross-fades between two drawn frames, so a 10 s clip costs
about as much as its 3-4 s build.

Usage
    python animate.py --list                                  # cards available this week
    python animate.py --card pick_rate_winrate_scatter        # one clip, default timing
    python animate.py --card pick_rate_winrate_scatter --build 4 --hold 2 \\
                      --highlight "Haze:3; Victor+Mirage:3; -:1"
    python animate.py --all                                   # every card, default timing
    python animate.py --scenes video_ep12.csv                 # a planned video, one clip per row
    python animate.py week12_analysis --all --codec prores    # editing-friendly ProRes .mov

Highlight beats: "Name:seconds" separated by ";". Join names with "+" to highlight several at
once, and use "-" to go back to the full chart. Archetype charts take archetype names
("marksman"), hero charts take hero names. Charts without highlight support ignore beats.

Scenes CSV (one clip per row; build/hold/highlight/name are optional):
    card,build,hold,highlight,name
    hero_winrates_top_bottom,3.5,2,,
    pick_rate_winrate_scatter,4,1.5,Haze:3; Victor+Mirage:3,scatter_haze_victor

Clips land in <week_dir>/videos/. Set ANIMATE_WORKERS to change the number of parallel
render processes (default: all CPU cores).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402

CHART_MODULES = [
    "hero_wr_total_matches_split_column", "pick_rate_winrate_scatter", "balance_wr_distribution",
    "archetype_stacking", "matchup_bubble_chart", "hero_strength_and_scaling", "match_duration",
    "hero_duo_synergy_table", "most_freq_comps", "comp_uniqueness", "week_over_week",
]
DEFAULT_BUILD_S = 4.0       # seconds for the card to draw itself in
DEFAULT_HOLD_S = 3.0        # seconds on the finished card (before any highlight beats)
CROSSFADE_S = 0.35          # highlight on/off transition
CODECS = {
    # small, plays everywhere (YouTube, phones, any editor)
    "h264": (".mp4", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                      "-movflags", "+faststart"]),
    # large, but frame-accurate and lossless-ish for scrubbing in an editor
    "prores": (".mov", ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]),
}


# --- SCENES ------------------------------------------------------------------------
@dataclass
class Beat:
    names: frozenset[str] | None     # None = back to the full chart
    seconds: float


@dataclass
class Scene:
    card: str
    build: float = DEFAULT_BUILD_S
    hold: float = DEFAULT_HOLD_S
    beats: list[Beat] = field(default_factory=list)
    name: str | None = None


def parse_beats(text: str | None) -> list[Beat]:
    beats = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        who, _, secs = part.rpartition(":")
        if not who:
            raise ValueError(f'highlight beat "{part}" needs a duration, e.g. "Haze:3"')
        names = None if who.strip() == "-" else frozenset(n.strip() for n in who.split("+") if n.strip())
        beats.append(Beat(names, float(secs)))
    return beats


def read_scenes(path: Path) -> list[Scene]:
    scenes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if not row.get("card") or row["card"].startswith("#"):
                continue
            scenes.append(Scene(card=row["card"],
                                build=float(row.get("build") or DEFAULT_BUILD_S),
                                hold=float(row.get("hold") or DEFAULT_HOLD_S),
                                beats=parse_beats(row.get("highlight")),
                                name=row.get("name") or None))
    return scenes


# --- CARDS -------------------------------------------------------------------------
def collect_cards(data_dir: Path, **opts) -> dict:
    """{card name: draw(progress, highlight)} across every chart script."""
    out = {}
    for mod_name in CHART_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            out.update(mod.cards(data_dir, **opts))
        except FileNotFoundError as e:          # this week lacks that script's CSV
            print(f"  skipping {mod_name}: {e.filename or e} not found")
    return out


# Worker processes inherit _CARDS through fork, so the week's data is loaded once, not per frame.
_CARDS: dict = {}


def _draw(job) -> bytes:
    card, progress, highlight, layout = job
    hl = set(highlight) if highlight else None
    return T.fig_to_array(_CARDS[card](progress, hl), layout).tobytes()


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)           # smoothstep: soft start and stop


# --- RENDER ------------------------------------------------------------------------
def render_scene(scene: Scene, pool, out_path: Path, fps: int, codec: str) -> float:
    draw = _CARDS[scene.card]
    layout = T.card_layout(draw(1.0, None))           # lock the finished card's layout
    n_build = max(2, round(scene.build * fps))
    build_jobs = [(scene.card, i / (n_build - 1), None, layout) for i in range(n_build)]
    beat_jobs = [(scene.card, 1.0, tuple(sorted(b.names)) if b.names else None, layout) for b in scene.beats]

    ext, enc = CODECS[codec]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{T.CARD_PX}x{T.CARD_PX}", "-r", str(fps), "-i", "-", *enc, str(out_path.with_suffix(ext))]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_frames = 0

    def write(frame: bytes, times: int = 1):
        nonlocal n_frames
        for _ in range(times):
            ff.stdin.write(frame)
        n_frames += times

    last = None
    for frame in pool.imap(_draw, build_jobs, chunksize=4):   # in order, while workers render ahead
        write(frame)
        last = frame
    full = last
    write(full, max(0, round(scene.hold * fps)))

    current = np.frombuffer(full, np.uint8)
    full_arr = current
    targets = pool.map(_draw, beat_jobs) if beat_jobs else []
    n_fade = max(1, round(CROSSFADE_S * fps))
    for beat, target in zip(scene.beats, targets):
        tgt = full_arr if beat.names is None else np.frombuffer(target, np.uint8)
        a = current.astype(np.float32)
        b = tgt.astype(np.float32)
        for i in range(1, n_fade + 1):
            w = ease(i / n_fade)
            write((a + (b - a) * w).round().astype(np.uint8).tobytes())
        write(tgt.tobytes(), max(0, round(beat.seconds * fps) - n_fade))
        current = tgt

    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path.name}")
    return n_frames / fps


def main() -> int:
    ap = argparse.ArgumentParser(description="Render chart cards as video clips")
    ap.add_argument("week_dir", nargs="?", default=None)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--list", action="store_true", help="list this week's cards")
    what.add_argument("--card", help="one card by name (see --list)")
    what.add_argument("--all", action="store_true", help="every card with default timing")
    what.add_argument("--scenes", type=Path, help="CSV of clips to render (see docstring)")
    ap.add_argument("--build", type=float, default=DEFAULT_BUILD_S, help="build-in seconds")
    ap.add_argument("--hold", type=float, default=DEFAULT_HOLD_S, help="seconds on the finished card")
    ap.add_argument("--highlight", default="", help='beats, e.g. "Haze:3; Victor+Mirage:3; -:1"')
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--codec", choices=list(CODECS), default="h264")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <week_dir>/videos")
    ap.add_argument("--compare", default=None, help="match_duration: folder of the period to compare")
    ap.add_argument("--compare-label", default="before the patch")
    ap.add_argument("--base", default=None, help="week_over_week: folder to compare against (default: last week)")
    args = ap.parse_args()

    data_dir = T.resolve_target_dir(args.week_dir)
    opts = {"compare": args.compare, "compare_label": args.compare_label, "base": args.base}
    _CARDS.update(collect_cards(data_dir, **opts))
    if args.list:
        print("\n".join(sorted(_CARDS)))
        return 0

    if args.scenes:
        scenes = read_scenes(args.scenes)
    elif args.card:
        scenes = [Scene(args.card, args.build, args.hold, parse_beats(args.highlight))]
    else:
        scenes = [Scene(c, args.build, args.hold) for c in _CARDS]
    unknown = [s.card for s in scenes if s.card not in _CARDS]
    if unknown:
        print(f"Unknown card(s): {', '.join(unknown)}. Run with --list to see this week's cards.")
        return 1

    out_dir = args.out_dir or data_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("ANIMATE_WORKERS", os.cpu_count() or 4))
    used: dict[str, int] = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for s in scenes:
            name = s.name or s.card
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:                      # same card twice in one scenes file
                name = f"{name}_{used[name]}"
            t = time.time()
            secs = render_scene(s, pool, out_dir / name, args.fps, args.codec)
            print(f"{name}{CODECS[args.codec][0]}: {secs:.1f}s clip in {time.time() - t:.1f}s")
    print(f"Clips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
