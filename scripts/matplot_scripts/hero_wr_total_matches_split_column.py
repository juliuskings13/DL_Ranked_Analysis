import sys
import re
from datetime import date
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION & PATH RESOLUTION ---
RANKED_START_DATE = date(2026, 7, 30)


def get_current_week_number() -> int:
    """Calculates the target week number (0-indexed offset for completed week)."""
    days_elapsed = (date.today() - RANKED_START_DATE).days
    return days_elapsed // 7


def resolve_target_dir() -> Path:
    """Gets target dir from CLI args or defaults to current week's analysis folder."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            return target

    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"


DATA_DIR = resolve_target_dir()
OUTPUT_DIR = DATA_DIR / "graphs"
CSV_FILE_PATH = DATA_DIR / "hero_winrates.csv"

# --- COLOR PALETTE & TYPE CONFIGURATION ---
TYPE_COLORS = {
    "brawler": (34, 197, 94, 255),    # #22c55e
    "marksman": (249, 115, 22, 255),  # #f97316
    "assassin": (6, 182, 212, 255),   # #06b6d4
    "mystic": (214, 31, 255, 255),    # #d61fff
}

DIM_ALPHA = 0.22  # Translucent background track fill for bars

BG_COLOR = (15, 23, 23, 255)
TEXT_COLOR = (244, 240, 226, 255)
TITLE_COLOR = (255, 208, 115, 255)
TRACK_BG = (42, 42, 42, 255)
SUBTITLE_COLOR = (180, 185, 195, 255)
SECTION_HEADER_COLOR = (220, 220, 225, 255)
HEADER_TEXT_COLOR = (160, 165, 175, 255)
FOOTER_COLOR = (120, 125, 135, 255)

# Image & Panel Dimensions
WIDTH = 1000
ROW_HEIGHT = 44
HEADER_HEIGHT = 110
PADDING = 5
FOOTER_HEIGHT = 30

# Column Layout Boundaries
COL_HERO_NAME = (20, 140)
COL_MATCHES_BAR = (140, 550)
COL_WINRATE_BAR = (570, 980)


def hex_to_rgba(hex_str: str, alpha: int = 255) -> tuple:
    hex_str = hex_str.lstrip("#")
    r, g, b = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


def get_centered_x(text, font, col_bounds, panel_offset=0):
    bbox = font.getbbox(str(text))
    text_width = bbox[2] - bbox[0]
    col_width = col_bounds[1] - col_bounds[0]
    return panel_offset + col_bounds[0] + (col_width - text_width) // 2


def render_split_chart(
    data_items: list, title: str, subtitle: str, output_path: Path, is_grouped: bool = False
):
    """
    Renders a 4-split column hero stats chart (Name, Matches Bar, Winrate Bar).
    Handles both flat sorted lists and grouped archetype lists.
    """
    # Calculate Total Canvas Height
    num_rows = 0
    num_group_headers = 0

    if is_grouped:
        for group in data_items:
            num_group_headers += 1
            num_rows += len(group["heroes"])
    else:
        num_rows = len(data_items)

    calculated_height = (
        HEADER_HEIGHT
        + (num_rows * ROW_HEIGHT)
        + (num_group_headers * 40)
        + PADDING
        + FOOTER_HEIGHT - 10
    )

    image = Image.new("RGBA", (WIDTH, calculated_height), BG_COLOR)
    draw = ImageDraw.Draw(image)

    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        section_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        font = header_font = title_font = subtitle_font = section_font = footer_font = ImageFont.load_default()

    # Title & Subtitle
    draw.text((PADDING, PADDING), title, font=title_font, fill=TITLE_COLOR)
    draw.text((PADDING, PADDING + 38), subtitle, font=subtitle_font, fill=SUBTITLE_COLOR)

    # Legend Header
    legend_x = PADDING
    legend_y = PADDING + 70
    for h_type, color_rgba in TYPE_COLORS.items():
        draw.rectangle([(legend_x, legend_y + 3), (legend_x + 14, legend_y + 17)], fill=color_rgba)
        draw.text((legend_x + 20, legend_y), h_type, font=header_font, fill=TEXT_COLOR)
        legend_x += 120

    # Column Headers
    header_y = PADDING + 95
    matches_header_x = COL_MATCHES_BAR[0] + 100
    winrate_header_x = COL_WINRATE_BAR[0]+ 100
    draw.text((matches_header_x, header_y), "TOTAL MATCHES", font=header_font, fill=TITLE_COLOR)
    draw.text((winrate_header_x, header_y), "WIN RATE", font=header_font, fill=TITLE_COLOR)

    draw.line([(PADDING, HEADER_HEIGHT - 12), (WIDTH - PADDING, HEADER_HEIGHT - 12)], fill=(60, 60, 65, 255), width=1)

    # Reference limits for relative bar scaling
    all_heroes = []
    if is_grouped:
        for g in data_items:
            all_heroes.extend(g["heroes"])
    else:
        all_heroes = data_items

    max_matches = max([h["total_matches"] for h in all_heroes]) if all_heroes else 1
    min_wr = min([h["winrate"] for h in all_heroes]) if all_heroes else 0
    max_wr = max([h["winrate"] for h in all_heroes]) if all_heroes else 100

    current_y = HEADER_HEIGHT

    # Helper to render individual row
    def draw_hero_row(hero, y_pos):
        # Use full solid color (no DIM_ALPHA)
        h_color = TYPE_COLORS.get(hero["hero_type"].lower(), TEXT_COLOR)

        text_y = y_pos + (ROW_HEIGHT // 2) - 10
        bar_y1 = y_pos + 8
        bar_y2 = y_pos + ROW_HEIGHT - 8

        # 1. Hero Name (Right-aligned)
        bbox = font.getbbox(hero["hero_name"])
        name_width = bbox[2] - bbox[0]
        name_x = COL_HERO_NAME[1] - name_width - 10
        draw.text((name_x, text_y), hero["hero_name"], font=font, fill=TITLE_COLOR)

        # 2. Total Matches Bar (Solid Fill)
        m_start_x, m_end_x = COL_MATCHES_BAR
        m_max_width = m_end_x - m_start_x
        m_fill_pct = hero["total_matches"] / max_matches
        m_bar_w = int(m_max_width * m_fill_pct)

        draw.rectangle([(m_start_x, bar_y1), (m_end_x, bar_y2)], fill=TRACK_BG)
        draw.rectangle([(m_start_x, bar_y1), (m_start_x + m_bar_w, bar_y2)], fill=h_color)

        # White text inside solid vibrant bar for max contrast
        m_val_str = f"{hero['total_matches']:,}"
        m_val_bbox = font.getbbox(m_val_str)
        m_val_w = m_val_bbox[2] - m_val_bbox[0]
        draw.text((m_start_x + m_bar_w - m_val_w - 8, text_y), m_val_str, font=font, fill=(255, 255, 255, 255), stroke_width=1,stroke_fill=(0, 0, 0, 220))

        # 3. Win Rate Bar (Solid Fill)
        w_start_x, w_end_x = COL_WINRATE_BAR
        w_max_width = w_end_x - w_start_x
        w_fill_pct = (hero["winrate"] - (min_wr - 2)) / (max_wr - (min_wr - 2)) if max_wr != min_wr else 0.5
        w_bar_w = int(w_max_width * max(0.05, min(1.0, w_fill_pct)))

        draw.rectangle([(w_start_x, bar_y1), (w_end_x, bar_y2)], fill=TRACK_BG)
        draw.rectangle([(w_start_x, bar_y1), (w_start_x + w_bar_w, bar_y2)], fill=h_color)

        w_val_str = f"{hero['winrate']:.2f}%"
        w_val_bbox = font.getbbox(w_val_str)
        w_val_w = w_val_bbox[2] - w_val_bbox[0]
        draw.text((w_start_x + w_bar_w - w_val_w - 8, text_y), w_val_str, font=font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 220))

    # Drawing Loop
    if is_grouped:
        for group in data_items:
            # Group Section Header
            draw.text((COL_HERO_NAME[1] - 80, current_y + 10), group["group_name"], font=section_font, fill=SECTION_HEADER_COLOR)
            current_y += 35

            for hero in group["heroes"]:
                draw_hero_row(hero, current_y)
                current_y += ROW_HEIGHT
            current_y += 10
    else:
        for hero in data_items:
            draw_hero_row(hero, current_y)
            current_y += ROW_HEIGHT

    # Footer
    footer_y = calculated_height - FOOTER_HEIGHT + 10
    draw.text((PADDING, footer_y), "Chart: @eReys13 • Source: Deadlock Community API", font=footer_font, fill=FOOTER_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Successfully generated: {output_path.name}")


def main():
    if not CSV_FILE_PATH.exists():
        print(f"Error: CSV not found at {CSV_FILE_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE_PATH)
    df.columns = df.columns.str.strip().str.lower()
    records = df.to_dict(orient="records")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # CHART 1: Flat - Descending by Win Rate
    # ---------------------------------------------------------
    flat_winrate = sorted(records, key=lambda x: x["winrate"], reverse=True)
    render_split_chart(
        data_items=flat_winrate,
        title="Hero Total Matches & Winrate (Ranked by Win Rate)",
        subtitle="Overall hero performance overview ranked from highest to lowest win rate.",
        output_path=OUTPUT_DIR / "hero_stats_flat_desc_winrate.png",
        is_grouped=False,
    )

    # ---------------------------------------------------------
    # CHART 2: Flat - Descending by Total Matches
    # ---------------------------------------------------------
    flat_matches = sorted(records, key=lambda x: x["total_matches"], reverse=True)
    render_split_chart(
        data_items=flat_matches,
        title="Hero Total Matches & Winrate (Ranked by Matches Played)",
        subtitle="Overall hero pick frequency ranked from highest to lowest total matches.",
        output_path=OUTPUT_DIR / "hero_stats_flat_desc_matches.png",
        is_grouped=False,
    )

    # Order archetypes logically
    archetypes = ["brawler", "mystic", "marksman", "assassin"]

    # ---------------------------------------------------------
    # CHART 3: Grouped by Archetype - Descending by Win Rate
    # ---------------------------------------------------------
    grouped_winrate = []
    for arch in archetypes:
        heroes = [r for r in records if str(r["hero_type"]).lower() == arch]
        heroes = sorted(heroes, key=lambda x: x["winrate"], reverse=True)
        if heroes:
            grouped_winrate.append({"group_name": arch, "heroes": heroes})

    render_split_chart(
        data_items=grouped_winrate,
        title="Hero Performance Grouped by Archetype (Desc. Win Rate)",
        subtitle="Comparing win rates within each archetype class.",
        output_path=OUTPUT_DIR / "hero_stats_grouped_desc_winrate.png",
        is_grouped=True,
    )

    # ---------------------------------------------------------
    # CHART 4: Grouped by Archetype - Descending by Total Matches
    # ---------------------------------------------------------
    grouped_matches = []
    for arch in archetypes:
        heroes = [r for r in records if str(r["hero_type"]).lower() == arch]
        heroes = sorted(heroes, key=lambda x: x["total_matches"], reverse=True)
        if heroes:
            grouped_matches.append({"group_name": arch, "heroes": heroes})

    render_split_chart(
        data_items=grouped_matches,
        title="Hero Performance Grouped by Archetype (Desc. Matches)",
        subtitle="Comparing pick volume within each archetype class.",
        output_path=OUTPUT_DIR / "hero_stats_grouped_desc_matches.png",
        is_grouped=True,
    )


if __name__ == "__main__":
    main()