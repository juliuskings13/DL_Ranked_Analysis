"""
animate.py — turn the weekly chart cards into video clips (T.OUT_PX square, 1440x1440 by default;
cards with their own shape, like hero_<name>_wide 2560x1440 and hero_<name>_tall 1440x2560, keep it).

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
    python animate.py --all                                   # every square card, default timing
    python animate.py --all --all-shapes                      # plus the wide / tall hero cards
    python animate.py --card hero_victor_wide                 # one 2560x1440 hero card
    python animate.py --card "hero_*_tall"                    # every tall (shorts) hero card, one clip each
    python animate.py --scenes video_ep12.csv                 # a planned video, one clip per row
    python animate.py week12_analysis --all --codec prores    # editing-friendly ProRes .mov
    python animate.py <folder> --scenes plans/hero_compendium_scenes.csv --concat hero_compendium.mp4 --fps 30
                                                              # every clip joined into one video (in videos/)

Encoding: HEVC on the GPU (hevc_vaapi, quality 20, mp4), same as OBS and render_pipeline, with a keyframe every
2 s. --codec h264 for the CPU encoder (used automatically if the GPU one isn't there).

Highlight beats: "Name:seconds" separated by ";". Join names with "+" to highlight several at
once, and use "-" to go back to the full chart. Archetype charts take archetype names
("marksman"), hero charts take hero names. Charts without highlight support ignore beats.

Scenes CSV (one clip per row; build/hold/highlight/name are optional):
    card,build,hold,highlight,name
    hero_winrates_top_bottom,3.5,2,,
    pick_rate_winrate_scatter,4,1.5,Haze:3; Victor+Mirage:3,scatter_haze_victor

Clips land in <week_dir>/videos/. --scenes paths are looked up in the week / period folder too. Set ANIMATE_WORKERS to change the number of parallel
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
    "hero_spotlight", "rank_performance", "patch_notes", "season_summary", "season_reset",
]
DEFAULT_BUILD_S = 6.0       # seconds for a chart to draw itself in (cards can suggest their own: draw.build_s)
DEFAULT_HOLD_S = 3.0        # seconds on the finished card (before any highlight beats)
CROSSFADE_S = 0.35          # highlight on/off transition
VAAPI_DEVICE = "/dev/dri/renderD128"
VAAPI_QUALITY = 20          # same as OBS (hevc_vaapi, CQP 20) and render_pipeline's masters
KEYFRAME_S = 2              # a keyframe every 2 s: outro.py can then re-encode only a video's last seconds
CODECS = {
    # default: the GPU encoder, same codec / quality as OBS and render_pipeline (HEVC, CQP 20, mp4).
    # Falls back to h264 (CPU) automatically if the GPU encoder isn't available.
    "hevc": (".mp4", ["-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi",
                      "-vglobal_quality", str(VAAPI_QUALITY), "-movflags", "+faststart"]),
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
    build: float | None = None      # None = the card's own suggested build time (or DEFAULT_BUILD_S)
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
                                build=float(row["build"]) if row.get("build") else None,
                                hold=float(row.get("hold") or DEFAULT_HOLD_S),
                                beats=parse_beats(row.get("highlight")),
                                name=row.get("name") or None))
    return scenes


# --- CARDS -------------------------------------------------------------------------
def collect_cards(data_dir: Path, **opts) -> dict:
    """{card name: draw(progress, highlight)} across every chart script."""
    out = {}
    # a patch folder (patch_notes.py) and a season folder (season_summary.py) only have their own cards
    # a folder can hold a season analysis (season_summary.py) next to a whole-season period's weekly-style
    # exports (e.g. season_1/ from both `season_summary.py export` and `run_pipeline.py --period Season_1`)
    d = Path(data_dir)
    if (d / "changes.csv").exists():
        # patch folder; with season_reset's two data files copied in, its cards join (combined video)
        modules = ["patch_notes"] + (["season_reset"] if (d / "season_reset_meta.json").exists() else [])
    else:
        modules = []
        if (d / "season_meta.json").exists():
            modules.append("season_summary")
        if (d / "season_reset_meta.json").exists():
            modules.append("season_reset")
        if (d / "hero_winrates.csv").exists() or not modules:
            modules += [m for m in CHART_MODULES if m not in ("patch_notes", "season_summary", "season_reset")]
    for mod_name in modules:
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
    finished = draw(1.0, None)
    width, height = int(round(finished.bbox.width)), int(round(finished.bbox.height))   # real pixels
    layout = T.card_layout(finished)                  # lock the finished card's layout
    T.plt.close(finished)                             # only needed for its size and layout
    build = scene.build if scene.build is not None else getattr(draw, "build_s", DEFAULT_BUILD_S)
    n_build = max(2, round(build * fps))
    build_jobs = [(scene.card, i / (n_build - 1), None, layout) for i in range(n_build)]
    beat_jobs = [(scene.card, 1.0, tuple(sorted(b.names)) if b.names else None, layout) for b in scene.beats]

    ext, enc = CODECS[codec]
    pre = ["-vaapi_device", VAAPI_DEVICE] if codec == "hevc" else []
    gop = ["-g", str(fps * KEYFRAME_S)] if codec in ("hevc", "h264") else []
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *pre, "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", *enc, *gop, str(out_path.with_suffix(ext))]
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
    what.add_argument("--card", help='one card by name (see --list), or a pattern for several, e.g. "hero_*_tall"')
    what.add_argument("--all", action="store_true", help="every card with default timing")
    what.add_argument("--scenes", type=Path, help="CSV of clips to render (see docstring)")
    ap.add_argument("--build", type=float, default=None,
                    help=f"build-in seconds (default: the card's own pace, e.g. ~1s per line on text cards, "
                         f"else {DEFAULT_BUILD_S:g}s)")
    ap.add_argument("--all-shapes", action="store_true",
                    help="with --all: also the wide (16:9) and tall (9:16) hero cards")
    ap.add_argument("--hold", type=float, default=DEFAULT_HOLD_S, help="seconds on the finished card")
    ap.add_argument("--highlight", default="", help='beats, e.g. "Haze:3; Victor+Mirage:3; -:1"')
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--concat", type=Path, default=None,
                    help="also join the rendered clips, in order, into this one video file")
    ap.add_argument("--codec", choices=list(CODECS), default="hevc",
                    help="hevc (default, GPU, same as OBS), h264 (CPU) or prores (editing)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <week_dir>/videos")
    ap.add_argument("--compare", default=None, help="match_duration: folder of the period to compare")
    ap.add_argument("--compare-label", default="before the patch")
    ap.add_argument("--base", default=None, help="week_over_week: folder to compare against (default: last week)")
    args = ap.parse_args()

    if args.codec == "hevc" and not vaapi_ok():
        print("  [i] GPU encoder (hevc_vaapi) not available here: using h264 (CPU) instead")
        args.codec = "h264"
    data_dir = T.resolve_target_dir(args.week_dir)
    opts = {"compare": args.compare, "compare_label": args.compare_label, "base": args.base}
    _CARDS.update(collect_cards(data_dir, **opts))
    if args.list:
        print("\n".join(sorted(_CARDS)))
        return 0

    if args.scenes:
        if not args.scenes.exists() and (data_dir / args.scenes).exists():
            args.scenes = data_dir / args.scenes          # plans/... inside the week / period folder
        if not args.scenes.exists():
            print(f"Scenes file not found: {args.scenes} (also looked in {data_dir})")
            return 1
        scenes = read_scenes(args.scenes)
    elif args.card and any(ch in args.card for ch in "*?["):      # a pattern: every matching card
        import fnmatch
        names = sorted(c for c in _CARDS if fnmatch.fnmatch(c, args.card))
        if not names:
            print(f"No card matches {args.card!r}. Run with --list to see this week's cards.")
            return 1
        scenes = [Scene(c, args.build, args.hold, parse_beats(args.highlight)) for c in names]
    elif args.card:
        scenes = [Scene(args.card, args.build, args.hold, parse_beats(args.highlight))]
    else:
        scenes = [Scene(c, args.build, args.hold) for c in _CARDS
                  if args.all_shapes or not c.endswith(("_wide", "_tall"))]
    unknown = [s.card for s in scenes if s.card not in _CARDS]
    if unknown:
        print(f"Unknown card(s): {', '.join(unknown)}. Run with --list to see this week's cards.")
        return 1

    out_dir = args.out_dir or data_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.environ.get("ANIMATE_WORKERS", os.cpu_count() or 4))
    used: dict[str, int] = {}
    rendered: list[Path] = []
    ext = CODECS[args.codec][0]
    with mp.get_context("fork").Pool(workers) as pool:
        for s in scenes:
            name = s.name or s.card
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:                      # same card twice in one scenes file
                name = f"{name}_{used[name]}"
            t = time.time()
            secs = render_scene(s, pool, out_dir / name, args.fps, args.codec)
            rendered.append((out_dir / name).with_suffix(ext))
            print(f"{name}{ext}: {secs:.1f}s clip in {time.time() - t:.1f}s")
    print(f"Clips in {out_dir}")
    if args.concat:
        target = args.concat if args.concat.suffix else args.concat.with_suffix(ext)
        if target.parent == Path("."):                   # a bare name lands next to the clips
            target = out_dir / target
        concat(rendered, target)
    return 0


def vaapi_ok() -> bool:
    """Can this machine encode with hevc_vaapi? (a tiny test encode)"""
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-vaapi_device", VAAPI_DEVICE, "-f", "lavfi",
                        "-i", "testsrc2=s=1280x720:d=0.2", "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi",
                        "-f", "null", "-"], capture_output=True, text=True)
    if r.returncode != 0:
        print("  [i] hevc_vaapi test encode failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    return r.returncode == 0


def concat(clips: list[Path], out: Path) -> None:
    """Join clips back to back without re-encoding (they share codec, size and frame rate)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
           "-movflags", "+faststart", str(out)]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("ffmpeg concat failed")
    listing.unlink()
    print(f"Joined {len(clips)} clips into {out}")


if __name__ == "__main__":
    sys.exit(main())