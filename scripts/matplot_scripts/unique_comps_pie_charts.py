import sys
import math
from datetime import date
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION & PATH RESOLUTION ---
TITLE = "Hero Comp Frequency Breakdown"
SUBTITLE = "Comparison of Unique Team Compositions vs. Total Matches Played"

RANKED_START_DATE = date(2026, 7, 30)

# 4x Supersampling (SSAA) for anti-aliasing
SCALE = 4

def get_current_week_number() -> int:
    days_elapsed = (date.today() - RANKED_START_DATE).days
    return max(0, days_elapsed // 7)

def resolve_target_dir() -> Path:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            return target

    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"

DATA_DIR = resolve_target_dir()
CSV_PATH = DATA_DIR / "unique_comp_freq_match.csv"
OUTPUT_DIR = DATA_DIR / "graphs"
OUTPUT_FILE = OUTPUT_DIR / "hero_comp_multipanel_pie.png"

# --- THEME COLORS ---
BG_COLOR = (15, 23, 23, 255)
TEXT_COLOR = (244, 240, 226, 255)
TITLE_COLOR = (255, 208, 115, 255)
SUBTITLE_COLOR = (180, 185, 195, 255)
FOOTER_COLOR = (140, 145, 155, 255)

SLICE_SINGLE = (36, 117, 98, 255)
SLICE_LOW = (46, 230, 147, 255)
SLICE_META = (235, 230, 204, 255)

# --- HELPER FUNCTIONS ---
def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    scaled_size = int(size * SCALE)
    font_names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]
    for name in font_names:
        try:
            return ImageFont.truetype(name, scaled_size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_outlined_text(draw: ImageDraw.ImageDraw, xy: tuple, text: str, font: ImageFont.ImageFont, 
                       fill: tuple, stroke_width: int = 1):
    scaled_xy = (xy[0] * SCALE, xy[1] * SCALE)
    scaled_stroke = int(stroke_width * SCALE)
    draw.text(scaled_xy, text, font=font, fill=fill, stroke_width=scaled_stroke, stroke_fill=(0, 0, 0, 255))

def get_text_width(font: ImageFont.ImageFont, text: str) -> float:
    bbox = font.getbbox(text)
    return (bbox[2] - bbox[0]) / SCALE

def prepare_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    
    single_comp = df.loc[df['frequency_bucket'] == 'single-occurrence', 'unique_team_comps'].values[0]
    low_comp = df.loc[df['frequency_bucket'] == '2 - 4 occurrences', 'unique_team_comps'].values[0]
    meta_comp = df.loc[~df['frequency_bucket'].isin(['single-occurrence', '2 - 4 occurrences']), 'unique_team_comps'].sum()

    single_games = df.loc[df['frequency_bucket'] == 'single-occurrence', 'total_games_played'].values[0]
    low_games = df.loc[df['frequency_bucket'] == '2 - 4 occurrences', 'total_games_played'].values[0]
    meta_games = df.loc[~df['frequency_bucket'].isin(['single-occurrence', '2 - 4 occurrences']), 'total_games_played'].sum()

    categories = [
        {"name": "Single-Occurrence (1 match)", "color": SLICE_SINGLE},
        {"name": "Low Frequency (2-4 matches)", "color": SLICE_LOW},
        {"name": "Meta Comps (5+ matches)", "color": SLICE_META}
    ]

    return categories, [single_comp, low_comp, meta_comp], [single_games, low_games, meta_games]

def clamp_x(x: float, text_width: float, base_width: float, padding: float) -> float:
    min_x = padding
    max_x = base_width - padding - text_width
    return max(min_x, min(x, max_x))

def draw_pie_panel(draw: ImageDraw.ImageDraw, center_x: float, center_y: float, radius: float, 
                   values: list, categories: list, panel_title: str,
                   title_font: ImageFont.ImageFont, label_font: ImageFont.ImageFont,
                   base_w: float, padding_x: float):
    t_width = get_text_width(title_font, panel_title)
    draw_outlined_text(draw, (center_x - t_width / 2, center_y - radius - 120), panel_title, title_font, TITLE_COLOR, stroke_width=1)

    total = sum(values)
    start_angle = -90

    cx_s, cy_s = center_x * SCALE, center_y * SCALE
    rad_s = radius * SCALE
    box = [cx_s - rad_s, cy_s - rad_s, cx_s + rad_s, cy_s + rad_s]

    for val, cat in zip(values, categories):
        sweep = (val / total) * 360
        end_angle = start_angle + sweep

        draw.pieslice(box, start=start_angle, end=end_angle, fill=cat["color"], outline=BG_COLOR, width=int(2 * SCALE))

        mid_angle_rad = math.radians(start_angle + sweep / 2)
        pct = (val / total) * 100
        
        px = center_x + (radius + 15) * math.cos(mid_angle_rad)
        py = center_y + (radius + 15) * math.sin(mid_angle_rad)
        
        arm_dir = 1 if math.cos(mid_angle_rad) >= 0 else -1
        lx = px + (35 * arm_dir)
        ly = py

        label_str = f"{cat['name']}"
        val_str = f"({pct:.1f}% | {int(val):,})"
        
        lbl_w = get_text_width(label_font, label_str)
        val_w = get_text_width(label_font, val_str)
        max_w = max(lbl_w, val_w)

        tx = lx + (10 if arm_dir == 1 else -10)
        unclamped_x = tx if arm_dir == 1 else tx - max_w
        
        # Prevent clipping beyond image borders
        final_x = clamp_x(unclamped_x, max_w, base_w, padding_x)
        final_lx = final_x - 10 if arm_dir == 1 else final_x + max_w + 10

        draw.line(
            [(px * SCALE, py * SCALE), (final_lx * SCALE, ly * SCALE)], 
            fill=SUBTITLE_COLOR, 
            width=int(2 * SCALE)
        )

        draw_outlined_text(draw, (final_x, ly - 20), label_str, label_font, TEXT_COLOR, stroke_width=1)
        draw_outlined_text(draw, (final_x, ly + 2), val_str, label_font, SUBTITLE_COLOR, stroke_width=1)

        start_angle = end_angle

# --- MAIN GENERATION ---
def generate_chart():
    BASE_W, BASE_H = 1800, 1000
    WIDTH, HEIGHT = BASE_W * SCALE, BASE_H * SCALE
    PADDING_X = 60

    hi_res_image = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(hi_res_image)

    title_font = load_font(36, bold=True)
    panel_title_font = load_font(26, bold=True)
    subtitle_font = load_font(20, bold=False)
    label_font = load_font(16, bold=True)
    footer_font = load_font(16, bold=False)

    # Header
    draw_outlined_text(draw, (PADDING_X, 40), TITLE, title_font, TITLE_COLOR, stroke_width=1)
    draw_outlined_text(draw, (PADDING_X, 90), SUBTITLE, subtitle_font, SUBTITLE_COLOR, stroke_width=1)

    categories, comp_values, game_values = prepare_data(CSV_PATH)

    # Panels
    radius = 200
    panel1_cx, panel2_cx = 450, 1250
    panel_cy = 520

    draw_pie_panel(draw, panel1_cx, panel_cy, radius, comp_values, categories, "By Unique Team Comps", panel_title_font, label_font, BASE_W, PADDING_X)
    draw_pie_panel(draw, panel2_cx, panel_cy, radius, game_values, categories, "By Total Matches Played", panel_title_font, label_font, BASE_W, PADDING_X)

    # Footer
    FOOTER_TOP = BASE_H - 55
    footer_y = BASE_H - 35
    draw.line(
        [(PADDING_X * SCALE, FOOTER_TOP * SCALE), ((BASE_W - PADDING_X) * SCALE, FOOTER_TOP * SCALE)], 
        fill=(40, 45, 50, 255), 
        width=int(1 * SCALE)
    )
    draw_outlined_text(draw, (PADDING_X, footer_y), "Data Source: Deadlock Community API", footer_font, FOOTER_COLOR, stroke_width=1)

    sig_text = "Analysis by @eReys13"
    sig_x = BASE_W - PADDING_X - get_text_width(footer_font, sig_text)
    draw_outlined_text(draw, (sig_x, footer_y), sig_text, footer_font, FOOTER_COLOR, stroke_width=1)

    # Resample with Lanczos for smooth antialiasing
    final_image = hi_res_image.resize((BASE_W, BASE_H), Image.Resampling.LANCZOS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_image.save(OUTPUT_FILE)
    print(f"Successfully generated bounded multipanel pie chart at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_chart()