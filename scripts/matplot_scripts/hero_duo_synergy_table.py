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
        "csv": DATA_DIR / "hero_duo_synergies.csv",
        "title": "TOP FREQUENT HERO DUOS",
        "output": OUTPUT_DIR / "top_hero_duos_table.png",
    }
]    

# --- DEADLOCK SCHEME CONFIGURATION ---
BG_COLOR = (15, 23, 23, 255)       # Solid deep teal-slate
STRIPE_COLOR = (22, 36, 34, 255)   # Alternating row tint
TEXT_COLOR = (244, 240, 226, 255)  # Cream text
TITLE_COLOR = (255, 208, 115, 255) # Deadlock amber gold accent
GREEN_WR = (65, 214, 126, 255)
RED_WR = (239, 99, 99, 255)
FOOTER_COLOR = (140, 145, 155, 255)
DIVIDER_COLOR = (45, 60, 60, 255)

# Expanded Dimensions for 2-Column Side-by-Side Layout
WIDTH = 1800
PANEL_WIDTH = 860
ROW_HEIGHT = 100        # Taller rows for bigger, cleaner icons
HEADER_HEIGHT = 140
PADDING = 25
FOOTER_HEIGHT = 45
ICON_SIZE = 85          # Increased icon size for 2-hero view

# Relative Sub-Column Offsets (within each 860px panel)
COL_RANK = (10, 70)
COL_FREQ = (70, 210)
COL_ICONS = (210, 630)
COL_WR = (630, 850)


def get_hero_icon(name: str, winrate: float) -> Image.Image:
    """Selects icon asset folder based on composition win rate."""
    if winrate > 50.0:
        asset_folder = ASSETS_BASE / "gloat_icons"
    elif winrate == 50.0:
        asset_folder = ASSETS_BASE / "base_icons"
    else:
        asset_folder = ASSETS_BASE / "critical"

    icon_path = asset_folder / f"{name}.png"
    if not icon_path.exists():
        icon_path = ASSETS_BASE / "base_icons" / f"{name}.png"

    if icon_path.exists():
        img = Image.open(icon_path).convert("RGBA")
        img.thumbnail((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
        return img
    else:
        return Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (60, 60, 65, 255))


def get_centered_x(text, font, col_bounds, panel_offset=0):
    """Calculates starting X pixel coordinate to center text within column bounds."""
    bbox = font.getbbox(str(text))
    text_width = bbox[2] - bbox[0]
    col_width = col_bounds[1] - col_bounds[0]
    return panel_offset + col_bounds[0] + (col_width - text_width) // 2


def render_table(data: list, title: str, output_path: Path):
    """Renders a 2-column side-by-side hero duo table from input data."""
    # Splitting top 20 into two columns of 10
    col1_data = data[:10]
    col2_data = data[10:20]
    num_rows = max(len(col1_data), len(col2_data))

    calculated_height = HEADER_HEIGHT + (num_rows * ROW_HEIGHT) + PADDING + FOOTER_HEIGHT
    image = Image.new("RGBA", (WIDTH, calculated_height), BG_COLOR)
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except IOError:
        font = header_font = title_font = footer_font = ImageFont.load_default()

    # Title
    title_x = get_centered_x(title, title_font, (0, WIDTH))
    draw.text((title_x, PADDING), title, font=title_font, fill=TITLE_COLOR)

    # Column Panel Offsets
    panel_offsets = [PADDING, WIDTH - PANEL_WIDTH - PADDING]

    # Draw Headers for both panels
    header_y = PADDING + 55
    headers = [
        ("#", COL_RANK),
        ("Matches", COL_FREQ),
        ("Hero Duo", COL_ICONS),
        ("Win Rate", COL_WR),
    ]

    for p_offset in panel_offsets:
        for text, bounds in headers:
            x = get_centered_x(text, header_font, bounds, panel_offset=p_offset)
            draw.text((x, header_y), text, font=header_font, fill=TEXT_COLOR)
        draw.line(
            [(p_offset, HEADER_HEIGHT - 10), (p_offset + PANEL_WIDTH, HEADER_HEIGHT - 10)],
            fill=(80, 85, 95, 255),
            width=2,
        )

    # Vertical Separator between left and right tables
    middle_x = WIDTH // 2
    draw.line([(middle_x, HEADER_HEIGHT - 20), (middle_x, calculated_height - FOOTER_HEIGHT - 10)], fill=DIVIDER_COLOR, width=3)

    # Render Left and Right Sub-Tables
    panels_data = [col1_data, col2_data]

    for panel_idx, panel_rows in enumerate(panels_data):
        p_offset = panel_offsets[panel_idx]

        # Alternating Row Backgrounds
        y_offset = HEADER_HEIGHT
        for i, _ in enumerate(panel_rows):
            fill_color = STRIPE_COLOR if i % 2 == 1 else BG_COLOR
            draw.rectangle([(p_offset, y_offset), (p_offset + PANEL_WIDTH, y_offset + ROW_HEIGHT)], fill=fill_color)
            y_offset += ROW_HEIGHT

        # Row Content
        y_offset = HEADER_HEIGHT
        for row in panel_rows:
            text_y = y_offset + (ROW_HEIGHT // 2) - 14
            icon_y = y_offset + (ROW_HEIGHT // 2) - (ICON_SIZE // 2)

            # Rank
            rank_str = str(row["rank"])
            rank_x = get_centered_x(rank_str, font, COL_RANK, p_offset)
            draw.text((rank_x, text_y), rank_str, font=font, fill=TEXT_COLOR)

            # Frequency
            freq_str = f"{row['freq']:,}"
            freq_x = get_centered_x(freq_str, font, COL_FREQ, p_offset)
            draw.text((freq_x, text_y), freq_str, font=font, fill=TEXT_COLOR)

            # Hero Icons (2 icons + generous gap)
            gap = 35
            total_icons_width = (2 * ICON_SIZE) + gap
            col_icons_width = COL_ICONS[1] - COL_ICONS[0]
            x_icon_offset = p_offset + COL_ICONS[0] + (col_icons_width - total_icons_width) // 2

            for hero in row["lineup"]:
                icon = get_hero_icon(hero, row["winrate"])
                image.paste(icon, (int(x_icon_offset), int(icon_y)), icon)
                x_icon_offset += ICON_SIZE + gap

            # Win Rate
            wr_str = f"{row['winrate']:.1f}%"
            wr_x = get_centered_x(wr_str, font, COL_WR, p_offset)
            wr_color = GREEN_WR if row["winrate"] >= 50.0 else RED_WR
            draw.text((wr_x, text_y), wr_str, font=font, fill=wr_color)

            y_offset += ROW_HEIGHT

    # Footer
    footer_y = calculated_height - FOOTER_HEIGHT + 10
    draw.text((30, footer_y), "Data Source: Deadlock Community API", font=footer_font, fill=FOOTER_COLOR)

    signature_text = "Analysis by @eReys13"
    sig_bbox = footer_font.getbbox(signature_text)
    sig_x = WIDTH - 30 - (sig_bbox[2] - sig_bbox[0])
    draw.text((sig_x, footer_y), signature_text, font=footer_font, fill=FOOTER_COLOR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Successfully generated 2-column duo table at: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for config in TABLE_CONFIGS:
        csv_path = config["csv"]
        if not csv_path.exists():
            print(f"Skipping {csv_path.name} (File not found)")
            continue

        df = pd.read_csv(csv_path)

        # 1. Rename column headers
        df = df.rename(
            columns={
                "matches_together": "freq",
                "duo_winrate": "winrate",
            }
        )

        # 2. Combine separate hero columns into a 2-element list
        df["lineup"] = df.apply(
            lambda row: [str(row["hero_name_a"]).strip(), str(row["hero_name_b"]).strip()],
            axis=1
        )

        # 3. Sort by match frequency and take top 20
        df = df.sort_values(by="freq", ascending=False).reset_index(drop=True)
        df = df.head(20).copy()

        # 4. Assign rank integer
        df["rank"] = df.index + 1

        data = df.to_dict(orient="records")
        render_table(data, config["title"], config["output"])


if __name__ == "__main__":
    main()