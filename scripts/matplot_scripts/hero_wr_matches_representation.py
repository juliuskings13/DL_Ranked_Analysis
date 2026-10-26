import sys
import math
from datetime import date
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION & PATH RESOLUTION ---
TITLE = "Hero Total Matches & Winrate"
SUBTITLE = "Highlighting under/over representation in relation to winrate"

RANKED_START_DATE = date(2026, 7, 30)


def get_current_week_number() -> int:
    days_elapsed = (date.today() - RANKED_START_DATE).days
    return days_elapsed // 7


def resolve_target_dir() -> Path:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            return target

    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"


DATA_DIR = resolve_target_dir()
CSV_PATH = DATA_DIR / "hero_winrates.csv"
OUTPUT_DIR = DATA_DIR / "graphs"
OUTPUT_FILE = OUTPUT_DIR / "hero_matches_winrate_horizontal_dumbbell.png"

# --- THEME COLORS ---
BG_COLOR = (15, 23, 23, 255)
TEXT_COLOR = (244, 240, 226, 255)
TITLE_COLOR = (255, 208, 115, 255)
SUBTITLE_COLOR = (180, 185, 195, 255)
FOOTER_COLOR = (140, 145, 155, 255)
GRID_COLOR = (30, 42, 42, 255)
STEM_COLOR = (100, 105, 115, 255)

COLOR_WINRATE = (34, 197, 94, 255)        # Vibrant Green (#22c55e)
COLOR_MATCHES = (6, 182, 212, 255)        # Cyan (#06b6d4)
COLOR_THRESHOLD_RED = (239, 68, 68, 255)  # Red (#ef4444) for <50% divider

TEXT_STROKE_COLOR = (10, 15, 15, 255)
TEXT_STROKE_WIDTH = 2

# Canvas Layout Dimensions
WIDTH = 1000
HEIGHT = 1150
PADDING_X = 20             # 20px outer margins

GRID_LEFT = 120            # 20px - 170px area for hero names
GRID_RIGHT = WIDTH - PADDING_X
GRID_WIDTH = GRID_RIGHT - GRID_LEFT

GRID_TOP = 170
FOOTER_TOP = HEIGHT - 35


def draw_outlined_text(draw, xy, text, font, fill, stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def draw_dashed_line(draw, start, end, fill, width=2, dash_length=8, space_length=5):
    """Draws a horizontal dashed line across the canvas."""
    x1, y = start
    x2, _ = end
    curr_x = x1
    while curr_x < x2:
        next_x = min(curr_x + dash_length, x2)
        draw.line([(curr_x, y), (next_x, y)], fill=fill, width=width)
        curr_x += dash_length + space_length


def generate_horizontal_dumbbell():
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # Sort ASCENDING by winrate
    df = df.sort_values(by="winrate", ascending=True).reset_index(drop=True)
    records = df.to_dict(orient="records")

    num_rows = len(records)

    # --- DYNAMIC 10K BOUNDS FOR TOTAL MATCHES AXIS ---
    raw_min_matches = df["total_matches"].min()
    raw_max_matches = df["total_matches"].max()

    axis_min_matches = math.floor(raw_min_matches / 10000) * 10000  # e.g., 14.8k -> 10,000
    axis_max_matches = math.ceil(raw_max_matches / 10000) * 10000   # e.g., 56.0k -> 60,000

    ticks = list(range(axis_min_matches, axis_max_matches + 10000, 10000))

    # Linear transformation helper mapping total match count to X coordinate
    def match_val_to_x(val):
        ratio = (val - axis_min_matches) / (axis_max_matches - axis_min_matches)
        return int(GRID_LEFT + (ratio * GRID_WIDTH))

    # Dynamic vertical row positioning to land right above footer
    bottom_row_target = FOOTER_TOP - 18
    row_height = (bottom_row_target - GRID_TOP) / (num_rows - 1)
    grid_bottom = bottom_row_target + 5

    image = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)

    # Load Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        value_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        hero_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except IOError:
        title_font = subtitle_font = label_font = value_font = hero_font = footer_font = ImageFont.load_default()

    # 1. Header
    draw_outlined_text(draw, (PADDING_X, 25), TITLE, title_font, TITLE_COLOR)
    draw_outlined_text(draw, (PADDING_X, 60), SUBTITLE, subtitle_font, SUBTITLE_COLOR)

    # 2. Legend (Updated label & prevents overlap)
    legend_items = [
        ("total_matches", COLOR_MATCHES), 
        ("win_rate (%)", COLOR_WINRATE),
        ("50% WR Threshold", COLOR_THRESHOLD_RED)
    ]
    leg_x = PADDING_X
    leg_y = 98
    dot_radius = 5
    for label, color in legend_items:
        tb = label_font.getbbox(label)
        item_w = tb[2] - tb[0]

        if "Threshold" in label:
            draw.line([(leg_x, leg_y + 6), (leg_x + 16, leg_y + 6)], fill=color, width=2)
            draw_outlined_text(draw, (leg_x + 22, leg_y), label, label_font, TEXT_COLOR)
            leg_x += item_w + 36
        else:
            draw.ellipse([(leg_x, leg_y + 2), (leg_x + (dot_radius * 2), leg_y + 2 + (dot_radius * 2))], fill=color)
            draw_outlined_text(draw, (leg_x + (dot_radius * 2) + 8, leg_y), label, label_font, TEXT_COLOR)
            leg_x += item_w + 36

    # 3. Background X Gridlines & 10k Ticks (10k, 20k, ... 60k)
    for tick in ticks:
        x_pos = match_val_to_x(tick)
        draw.line([(x_pos, GRID_TOP - 20), (x_pos, grid_bottom)], fill=GRID_COLOR, width=1)
        
        # Tick Header Label (e.g., "10k", "20k", "60k")
        t_str = f"{tick // 1000}k"
        tb = label_font.getbbox(t_str)
        t_w = tb[2] - tb[0]
        
        # Clamp tick label position inside canvas margins
        label_x = x_pos - t_w // 2
        label_x = max(PADDING_X, min(label_x, WIDTH - PADDING_X - t_w))
        draw_outlined_text(draw, (label_x, GRID_TOP - 22), t_str, label_font, FOOTER_COLOR, stroke_width=1)

    # 4. Win Rate Divider Index (< 50% vs >= 50%)
    divider_idx = None
    for idx, hero in enumerate(records):
        if hero["winrate"] >= 50.0:
            divider_idx = idx
            break

    # Render Horizontal Threshold Line
    if divider_idx is not None and divider_idx > 0:
        divider_y = int(GRID_TOP + (divider_idx * row_height) - (row_height / 2))
        draw_dashed_line(
            draw, 
            (GRID_LEFT, divider_y), 
            (GRID_RIGHT, divider_y), 
            fill=COLOR_THRESHOLD_RED, 
            width=2
        )
        draw_outlined_text(
            draw, 
            (PADDING_X, divider_y - 8), 
            "50% WR", 
            value_font, 
            COLOR_THRESHOLD_RED, 
            stroke_width=1
        )

    # 5. Render Dumbbell Rows
    node_radius = 5

    for idx, hero in enumerate(records):
        y_pos = int(GRID_TOP + (idx * row_height))

        matches_val = hero["total_matches"]
        winrate_val = hero["winrate"]

        # Winrate mapped across standard 0-100% scale relative to total match axis width
        x_matches = match_val_to_x(matches_val)
        
        # Winrate percentage mapped linearly across grid area (0% -> GRID_LEFT, 100% -> GRID_RIGHT)
        x_winrate = int(GRID_LEFT + ((winrate_val / 100.0) * GRID_WIDTH))

        # Row Guide Line across plot grid
        draw.line([(GRID_LEFT, y_pos), (GRID_RIGHT, y_pos)], fill=(20, 28, 28, 255), width=1)

        # Hero Name: Right-aligned to GRID_LEFT - 12
        name_str = hero["hero_name"]
        nb = hero_font.getbbox(name_str)
        name_w = nb[2] - nb[0]
        draw_outlined_text(draw, (GRID_LEFT - name_w - 12, y_pos - 7), name_str, hero_font, TEXT_COLOR)

        # Stem Barbell Connector
        draw.line([(x_matches, y_pos), (x_winrate, y_pos)], fill=STEM_COLOR, width=2)

        # Endpoint Circles
        draw.ellipse([(x_matches - node_radius, y_pos - node_radius), (x_matches + node_radius, y_pos + node_radius)], fill=COLOR_MATCHES)
        draw.ellipse([(x_winrate - node_radius, y_pos - node_radius), (x_winrate + node_radius, y_pos + node_radius)], fill=COLOR_WINRATE)

        # Formatted Metric Labels
        win_str = f"{winrate_val:.2f}%"
        matches_str = f"{matches_val / 1000:.1f}k"

        w_w = value_font.getbbox(win_str)[2] - value_font.getbbox(win_str)[0]
        m_w = value_font.getbbox(matches_str)[2] - value_font.getbbox(matches_str)[0]

        if x_winrate >= x_matches:
            w_x = x_winrate + 8
            m_x = x_matches - m_w - 8
        else:
            w_x = x_winrate - w_w - 8
            m_x = x_matches + 8

        draw_outlined_text(draw, (m_x, y_pos - 6), matches_str, value_font, COLOR_MATCHES, stroke_width=1)
        draw_outlined_text(draw, (w_x, y_pos - 6), win_str, value_font, COLOR_WINRATE, stroke_width=1)

    # 6. Compact Footer
    footer_y = HEIGHT - 22
    draw.line([(PADDING_X, FOOTER_TOP), (WIDTH - PADDING_X, FOOTER_TOP)], fill=(40, 45, 50, 255), width=1)
    draw_outlined_text(draw, (PADDING_X, footer_y), "Data Source: Deadlock Community API", footer_font, FOOTER_COLOR, stroke_width=1)

    sig_text = "Analysis by @eReys13"
    sig_x = WIDTH - PADDING_X - (footer_font.getbbox(sig_text)[2] - footer_font.getbbox(sig_text)[0])
    draw_outlined_text(draw, (sig_x, footer_y), sig_text, footer_font, FOOTER_COLOR, stroke_width=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_FILE)
    print(f"Successfully generated wide 10k-tick dumbbell chart at: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_horizontal_dumbbell()