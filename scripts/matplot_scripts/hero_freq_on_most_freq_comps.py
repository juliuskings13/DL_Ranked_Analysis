import sys
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
ASSETS_BASE = Path.home() / "Projects" / "DL_ranked_analysis" / "assets"

TABLE_CONFIGS = [
    {
        "csv": DATA_DIR / "hero_freq_top_comps.csv",
        "title": "HEROES ON MOST FREQUENT COMPS",
        "output": OUTPUT_DIR / "most_freq_hero_freq_team_comps.png",
    },
    {
        "csv": DATA_DIR / "hero_freq_top_win_comps.csv",
        "title": "HEROES ON MOST FREQUENT WINNING COMPS",
        "output": OUTPUT_DIR / "most_freq_hero_freq_winning_team_comps_table.png",
    },
    {
        "csv": DATA_DIR / "hero_freq_top_lose_comps.csv",
        "title": "HEROES ON MOST FREQUENT LOSING COMPS",
        "output": OUTPUT_DIR / "most_freq_hero_freq_losing_team_comps_table.png",
    },
]

# --- DEADLOCK SCHEME CONFIGURATION ---
BG_COLOR = (15, 23, 23, 255)          # Solid deep teal-slate
STRIPE_COLOR = (22, 36, 34, 255)      # Alternating row tint
TEXT_COLOR = (244, 240, 226, 255)     # Cream logo text
TITLE_COLOR = (255, 208, 115, 255)    # Amber gold accent
FOOTER_COLOR = (140, 145, 155, 255)   
BAR_COLOR = (255, 184, 70, 255)       # Solid Amber Accent Gold
DIVIDER_COLOR = (45, 60, 60, 255)

TEXT_STROKE_COLOR = (10, 15, 15, 255) 
TEXT_STROKE_WIDTH = 2                 

TYPE_COLORS = {
    "assassin": (239, 99, 99, 255),    
    "marksman": (255, 185, 70, 255),   
    "mystic": (168, 106, 255, 255),    
    "brawler": (65, 214, 126, 255)     
}

# Table Dimensions for 2-Column Side-by-Side View
WIDTH = 1800
PANEL_WIDTH = 860
ROW_HEIGHT = 80
HEADER_HEIGHT = 140
PADDING = 25
FOOTER_HEIGHT = 45

# Column Boundaries (relative within each panel)
COL_RANK = (10, 70)
COL_HERO = (70, 420)
COL_FREQ = (420, 640)
COL_TYPE = (640, 850)


def get_hero_icon(name: str, title: str) -> Image.Image:
    """Selects icon asset folder based on composition win rate."""
    if title == "HEROES ON MOST FREQUENT WINNING COMPS":
        asset_folder = ASSETS_BASE / "gloat_icons"
    elif title == "HEROES ON MOST FREQUENT LOSING COMPS":
        asset_folder = ASSETS_BASE / "critical"
    else:
        asset_folder = ASSETS_BASE / "base_icons"

    icon_path = asset_folder / f"{name}.png"
    if not icon_path.exists():
        icon_path = ASSETS_BASE / "base_icons" / f"{name}.png"

    icon_size = ROW_HEIGHT - 15
    if icon_path.exists():
        img = Image.open(icon_path).convert("RGBA")
        img.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
        return img
    else:
        return Image.new("RGBA", (icon_size, icon_size), (60, 60, 65, 255))


def get_centered_x(text, font, col_bounds, panel_offset=0):
    """Calculates starting X pixel coordinate to center text within column bounds."""
    bbox = font.getbbox(str(text))
    text_width = bbox[2] - bbox[0]
    col_width = col_bounds[1] - col_bounds[0]
    return panel_offset + col_bounds[0] + (col_width - text_width) // 2


def draw_outlined_text(draw, xy, text, font, fill, stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR):
    """Helper to draw text with a sharp dark outline."""
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def render_table(data: list, title: str, output_path: Path):
    """Renders a 2-column hero frequency table with inline bar graphics."""
    col1_data = data[:10]
    col2_data = data[10:20]
    num_rows = max(len(col1_data), len(col2_data))

    calculated_height = HEADER_HEIGHT + (num_rows * ROW_HEIGHT) + PADDING + FOOTER_HEIGHT
    image = Image.new("RGBA", (WIDTH, calculated_height), BG_COLOR)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        font = header_font = title_font = footer_font = ImageFont.load_default()

    # Title
    title_x = get_centered_x(title, title_font, (0, WIDTH))
    draw_outlined_text(draw, (title_x, PADDING), title, title_font, TITLE_COLOR)

    # Panel Offsets
    panel_offsets = [PADDING, WIDTH - PANEL_WIDTH - PADDING]

    # Headers for both panels
    header_y = PADDING + 55
    headers = [
        ("#", COL_RANK),
        ("Hero", COL_HERO),
        ("Frequency", COL_FREQ),
        ("Archetype", COL_TYPE)
    ]

    for p_offset in panel_offsets:
        for text, bounds in headers:
            x = get_centered_x(text, header_font, bounds, panel_offset=p_offset)
            draw_outlined_text(draw, (x, header_y), text, header_font, TEXT_COLOR)
        draw.line(
            [(p_offset, HEADER_HEIGHT - 10), (p_offset + PANEL_WIDTH, HEADER_HEIGHT - 10)],
            fill=(80, 85, 95, 255),
            width=2,
        )

    # Middle Separator
    middle_x = WIDTH // 2
    draw.line([(middle_x, HEADER_HEIGHT - 20), (middle_x, calculated_height - FOOTER_HEIGHT - 10)], fill=DIVIDER_COLOR, width=3)

    # Max frequency reference across whole dataset for relative bar scale
    max_freq = data[0]["hero_freq"] if data else 1  

    # Render Both Panels
    panels_data = [col1_data, col2_data]

    for panel_idx, panel_rows in enumerate(panels_data):
        p_offset = panel_offsets[panel_idx]

        # Alternating Backgrounds
        y_offset = HEADER_HEIGHT
        for i, _ in enumerate(panel_rows):
            fill_color = STRIPE_COLOR if i % 2 == 1 else BG_COLOR
            draw.rectangle([(p_offset, y_offset), (p_offset + PANEL_WIDTH, y_offset + ROW_HEIGHT)], fill=fill_color)
            y_offset += ROW_HEIGHT

        # Row Contents
        y_offset = HEADER_HEIGHT
        for row in panel_rows:
            text_y = y_offset + (ROW_HEIGHT // 2) - 13
            icon_dim = ROW_HEIGHT - 15
            icon_y = y_offset + (ROW_HEIGHT // 2) - (icon_dim // 2)

            # Rank
            rank_str = str(row["rank"])
            rank_x = get_centered_x(rank_str, font, COL_RANK, p_offset)
            draw_outlined_text(draw, (rank_x, text_y), rank_str, font, TEXT_COLOR)

            # Hero Icon + Name
            hero_icon = get_hero_icon(row["hero_name"], title)
            name_bbox = font.getbbox(row["hero_name"])
            name_width = name_bbox[2] - name_bbox[0]
            
            spacing = 15
            total_hero_width = icon_dim + spacing + name_width
            col_hero_width = COL_HERO[1] - COL_HERO[0]
            start_x = p_offset + COL_HERO[0] + (col_hero_width - total_hero_width) // 2

            image.paste(hero_icon, (start_x, icon_y), hero_icon)
            draw_outlined_text(draw, (start_x + icon_dim + spacing, text_y), row["hero_name"], font, TEXT_COLOR)

            # Inline Data Bar for Frequency
            freq_val = row["hero_freq"]
            fill_pct = freq_val / max_freq if max_freq else 0

            pad_x = 15
            bar_start_x = p_offset + COL_FREQ[0] + pad_x
            bar_max_width = (COL_FREQ[1] - COL_FREQ[0]) - (pad_x * 2)
            current_bar_width = int(bar_max_width * fill_pct)

            bar_y1 = text_y - 2
            bar_y2 = text_y + 24
            draw.rectangle(
                [(bar_start_x, bar_y1), (bar_start_x + current_bar_width, bar_y2)], 
                fill=BAR_COLOR
            )

            # Frequency Text
            freq_str = f"{freq_val:,}"
            freq_x = get_centered_x(freq_str, font, COL_FREQ, p_offset)
            draw_outlined_text(draw, (freq_x, text_y), freq_str, font, TEXT_COLOR)

            # Hero Archetype
            type_str = str(row["hero_type"]).capitalize()
            type_x = get_centered_x(type_str, font, COL_TYPE, p_offset)
            type_color = TYPE_COLORS.get(str(row["hero_type"]).lower(), TEXT_COLOR)
            draw_outlined_text(draw, (type_x, text_y), type_str, font, type_color)

            y_offset += ROW_HEIGHT

    # Footer
    footer_y = calculated_height - FOOTER_HEIGHT + 10
    draw_outlined_text(draw, (30, footer_y), "Data Source: Deadlock Community API", footer_font, FOOTER_COLOR, stroke_width=1)

    signature_text = "Analysis by @eReys13"
    sig_bbox = footer_font.getbbox(signature_text)
    sig_x = WIDTH - 30 - (sig_bbox[2] - sig_bbox[0])
    draw_outlined_text(draw, (sig_x, footer_y), signature_text, footer_font, FOOTER_COLOR, stroke_width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Successfully generated 2-column hero frequency table at: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for config in TABLE_CONFIGS:
        csv_path = config["csv"]
        if not csv_path.exists():
            print(f"Skipping {csv_path.name} (File not found)")
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()

        col_map = {
            "hero_name": "hero_name",
            "hero": "hero_name",
            "name": "hero_name",
            "hero_type": "hero_type",
            "type": "hero_type",
            "archetype": "hero_type",
            "frequency": "hero_freq",
            "freq": "hero_freq",
            "count": "hero_freq"
        }
        df = df.rename(columns=col_map)

        df = df.sort_values(by="hero_freq", ascending=False).reset_index(drop=True)
        df = df.head(20).copy()
        df["rank"] = df.index + 1

        data = df.to_dict(orient="records")
        render_table(data, config["title"], config["output"])


if __name__ == "__main__":
    main()