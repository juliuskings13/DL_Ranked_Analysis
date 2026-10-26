import os
import sys
from datetime import date
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import csv

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

# Archetype categories to plot along with their CSV data files
CATEGORIES = [
    {"key": "marksmen", "csv": "marksman_comp_wr.csv", "name": "MARKSMEN"},
    {"key": "assassin", "csv": "assassin_comp_wr.csv", "name": "ASSASSIN"},
    {"key": "brawler",  "csv": "brawler_comp_wr.csv", "name": "BRAWLER"},
    {"key": "mystic",   "csv": "mystic_comp_wr.csv", "name": "MYSTIC"},
]

# --- DEADLOCK SCHEME CONFIGURATION ---
BG_COLOR = (15, 23, 23, 255)          # Deep teal-slate canvas
TEXT_COLOR = (244, 240, 226, 255)     # Cream text
TITLE_COLOR = (255, 208, 115, 255)    # Amber gold accent
FOOTER_COLOR = (140, 145, 155, 255)   
GRID_COLOR = (25, 35, 35, 255)        # Soft grid line
BASELINE_COLOR = (60, 70, 75, 255)    # 50% threshold reference lines

GRADIENT_START = (255, 0, 0, 255)     # Crimson Red (Liability End)
GRADIENT_MID = (255, 255, 255, 255)   # Slate Gray (Neutral Midpoint)
GRADIENT_END = (0, 255, 0, 255)       # Emerald Green (Advantage End)

TEXT_STROKE_COLOR = (10, 15, 15, 255)
TEXT_STROKE_WIDTH = 2

# Canvas Dimensions
WIDTH = 1350 
HEIGHT = 1500
PADDING = 40

# --- LAYOUT: GRID AREA & LEGEND ---
LEGEND_HEADROOM = 160  
GRID_BOTTOM = 1230 
GRID_TOP = 155
GRID_LEFT = 145
GRID_RIGHT = 1205
GRID_HEIGHT = GRID_BOTTOM - GRID_TOP
GRID_WIDTH = GRID_RIGHT - GRID_LEFT

MAX_BUBBLE_RADIUS = 78

# --- HELPER FUNCTIONS ---
def load_csv_data(filepath: Path):
    """Loads CSV data in the format: t0_count, t1_count, total_matchups, t0_winrate."""
    raw_data = []
    if not filepath.exists():
        print(f"Warning: File {filepath} not found. Skipping...")
        return raw_data

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if not row or len(row) < 4:
                continue
            t0 = int(float(row[0]))
            t1 = int(float(row[1]))
            instances = int(float(row[2]))
            winrate = float(row[3])
            raw_data.append((t0, t1, instances, winrate))
    return raw_data

def get_centered_x(text, font, bounds):
    bbox = font.getbbox(str(text))
    return bounds[0] + ((bounds[1] - bounds[0]) - (bbox[2] - bbox[0])) // 2

def draw_outlined_text(draw, xy, text, font, fill, stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

def draw_bubble_label(draw, xy, text, font):
    x, y = xy
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, 190), stroke_width=2, stroke_fill=(0, 0, 0, 210))
    draw.text((x, y), text, font=font, fill=TEXT_COLOR, stroke_width=2, stroke_fill=(8, 12, 12, 255))

def count_to_x(count, min_count, max_count):
    ratio = (count - min_count) / (max_count - min_count) if max_count > min_count else 0
    usable_left = GRID_LEFT + MAX_BUBBLE_RADIUS
    usable_right = GRID_RIGHT - MAX_BUBBLE_RADIUS
    return int(usable_left + ratio * (usable_right - usable_left))

def count_to_y(count, min_count, max_count):
    ratio = (count - min_count) / (max_count - min_count) if max_count > min_count else 0
    usable_top = GRID_TOP + MAX_BUBBLE_RADIUS
    usable_bottom = GRID_BOTTOM - MAX_BUBBLE_RADIUS
    return int(usable_bottom - ratio * (usable_bottom - usable_top))

def get_bubble_size(instances, min_instances, max_instances):
    min_radius = 14
    max_radius = MAX_BUBBLE_RADIUS 
    if instances <= min_instances: return min_radius
    sqrt_ratio = math.sqrt(instances - min_instances) / math.sqrt(max_instances - min_instances) if max_instances > min_instances else 0
    return int(min_radius + (sqrt_ratio * (max_radius - min_radius)))

def get_bubble_color(winrate, opacity=230):
    deviation = winrate - 50.0
    max_diagnostic_deviation = 4.0

    intensity_ratio = abs(deviation) / max_diagnostic_deviation
    intensity_ratio = min(1, math.pow(intensity_ratio, 0.6))
    
    if deviation >= 0:
        r = int(GRADIENT_MID[0] + intensity_ratio * (GRADIENT_END[0] - GRADIENT_MID[0]))
        g = int(GRADIENT_MID[1] + intensity_ratio * (GRADIENT_END[1] - GRADIENT_MID[1]))
        b = int(GRADIENT_MID[2] + intensity_ratio * (GRADIENT_END[2] - GRADIENT_MID[2]))
    else:
        r = int(GRADIENT_MID[0] + intensity_ratio * (GRADIENT_START[0] - GRADIENT_MID[0]))
        g = int(GRADIENT_MID[1] + intensity_ratio * (GRADIENT_START[1] - GRADIENT_MID[1]))
        b = int(GRADIENT_MID[2] + intensity_ratio * (GRADIENT_START[2] - GRADIENT_MID[2]))

    return (r, g, b, opacity)

# --- MAIN GENERATION ENGINE ---
def generate_category_chart(cat_info):
    csv_file_path = DATA_DIR / cat_info["csv"]
    raw_data = load_csv_data(csv_file_path)
    if not raw_data:
        return

    output_file = OUTPUT_DIR / f"{cat_info['key']}_composition_bubble_scatter_complete.png"
    title = f"{cat_info['name']} QUANTITY MATCHUP WINRATE"
    subtitle = f"Win Rate Deviation vs. {cat_info['name'].capitalize()} Quantity Matchup"

    # Dynamic limits determined directly from dataset
    max_hero_count = max(max(d[0] for d in raw_data), max(d[1] for d in raw_data))
    min_hero_count = min(min(d[0] for d in raw_data), min(d[1] for d in raw_data))
    min_instances = min(d[2] for d in raw_data)
    max_instances = max(d[2] for d in raw_data)

    image = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    
    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        match_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        title_font = sub_font = label_font = match_font = footer_font = ImageFont.load_default()

    # 1. Header
    t_x = get_centered_x(title, title_font, (0, WIDTH))
    draw_outlined_text(draw, (t_x, PADDING + 10), title, title_font, TITLE_COLOR)
    s_x = get_centered_x(subtitle, sub_font, (0, WIDTH))
    draw_outlined_text(draw, (s_x, PADDING + 60), subtitle, sub_font, FOOTER_COLOR)

    # 2. Grid & Axes
    draw.rectangle([(GRID_LEFT, GRID_TOP), (GRID_RIGHT, GRID_BOTTOM)], outline=(60, 70, 75, 255), width=2)
    
    ticks = list(range(min_hero_count, max_hero_count + 1))
    for tick in ticks:
        x_pt = count_to_x(tick, min_hero_count, max_hero_count)
        y_pt = count_to_y(tick, min_hero_count, max_hero_count)
        
        draw.line([(x_pt, GRID_TOP), (x_pt, GRID_BOTTOM)], fill=GRID_COLOR, width=1)
        draw.line([(GRID_LEFT, y_pt), (GRID_RIGHT, y_pt)], fill=GRID_COLOR, width=1)
        
        t_l_color = BASELINE_COLOR if tick == 0 else TEXT_COLOR
        draw.line([(x_pt, GRID_BOTTOM), (x_pt, GRID_BOTTOM + 8)], fill=t_l_color, width=2)
        draw_outlined_text(draw, (x_pt - 8, GRID_BOTTOM + 15), str(tick), label_font, t_l_color)
        
        draw.line([(GRID_LEFT, y_pt), (GRID_LEFT - 8, y_pt)], fill=t_l_color, width=2)
        draw_outlined_text(draw, (GRID_LEFT - 35, y_pt - 10), str(tick), label_font, t_l_color)

    # Axis Labels
    x_label = f"Team 0 {cat_info['name'].capitalize()} Count"
    x_bbox = sub_font.getbbox(x_label)
    x_label_x = GRID_LEFT + (GRID_WIDTH - (x_bbox[2] - x_bbox[0])) // 2
    draw_outlined_text(draw, (x_label_x, GRID_BOTTOM + 38), x_label, sub_font, TITLE_COLOR, stroke_width=1)

    y_lbl_text = f"Team 1 {cat_info['name'].capitalize()} Count"
    y_bbox = sub_font.getbbox(y_lbl_text)
    y_lbl_img = Image.new("RGBA", (int(y_bbox[2] - y_bbox[0] + 10), int(y_bbox[3] - y_bbox[1] + 10)), (0, 0, 0, 0))
    y_lbl_draw = ImageDraw.Draw(y_lbl_img)
    y_lbl_draw.text((5, 5), y_lbl_text, font=sub_font, fill=TITLE_COLOR, stroke_width=1, stroke_fill=TEXT_STROKE_COLOR)
    y_lbl_rot = y_lbl_img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    y_label_x = max(10, GRID_LEFT - 75 - y_lbl_rot.width)
    y_label_y = GRID_TOP + (GRID_HEIGHT - y_lbl_rot.height) // 2
    image.alpha_composite(y_lbl_rot, (y_label_x, y_label_y))

    # 3. Render Matchup Bubbles
    bubble_records = []
    for t0_m, t1_m, instances, winrate in raw_data:
        x_pt = count_to_x(t0_m, min_hero_count, max_hero_count)
        y_pt = count_to_y(t1_m, min_hero_count, max_hero_count)
        radius = get_bubble_size(instances, min_instances, max_instances)
        color = get_bubble_color(winrate)
        bubble_records.append((t0_m, t1_m, instances, winrate, x_pt, y_pt, radius, color))

    # Draw smaller bubbles on top of large ones
    for _, _, _, _, x_pt, y_pt, radius, color in sorted(bubble_records, key=lambda r: r[6], reverse=True):
        draw.ellipse([(x_pt - radius, y_pt - radius), (x_pt + radius, y_pt + radius)], fill=color)

    # Render Bubble Labels
    for _, _, instances, _, x_pt, y_pt, radius, _ in bubble_records:
        instance_text = f"{instances:,}"

        if radius >= 58: label_size = 22
        elif radius >= 42: label_size = 19
        elif radius >= 30: label_size = 16
        elif radius >= 22: label_size = 13
        else: label_size = 11

        try:
            instance_font = ImageFont.truetype("DejaVuSansCondensed-Bold.ttf", label_size)
        except IOError:
            instance_font = match_font

        bbox = instance_font.getbbox(instance_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = x_pt - text_w // 2
        label_y = y_pt - text_h // 2 - bbox[1]

        draw_bubble_label(draw, (label_x, label_y), instance_text, instance_font)

    # 4. Legend
    legend_top_y = GRID_BOTTOM + 115 
    draw_outlined_text(draw, (GRID_LEFT, legend_top_y), "Team 0 Win Rate (50% Equilibrium)", match_font, TEXT_COLOR)
    
    bar_x1 = GRID_LEFT + 390
    bar_w = 440
    bar_h = 24
    bar_y1 = legend_top_y - 2
    
    bar_img = Image.new("RGBA", (bar_w, bar_h), (0,0,0,0))
    bar_draw = ImageDraw.Draw(bar_img)
    
    for x in range(bar_w):
        ratio = x / (bar_w - 1)
        cur_wr = 46.0 + (ratio * 8.0)
        cur_color = get_bubble_color(cur_wr, opacity=255)
        bar_draw.line([(x, 0), (x, bar_h)], fill=cur_color, width=1)

    image.alpha_composite(bar_img, (bar_x1, bar_y1))
    
    leg_steps = [46.0, 48.0, 50.0, 52.0, 54.0]
    l_sw = bar_w // (len(leg_steps) - 1)
    l_x = bar_x1
    
    for step in leg_steps:
        tick_l = 10
        tick_y_top = bar_y1 + bar_h + 3
        tick_color = BASELINE_COLOR if step == 50.0 else GRID_COLOR
        draw.line([(l_x, tick_y_top), (l_x, tick_y_top + tick_l)], fill=tick_color, width=1)
        
        t_str = f"{step:.1f}%"
        is_neutral = (step == 50.0)
        l_t_x = l_x - (match_font.getbbox(t_str)[2] - match_font.getbbox(t_str)[0]) // 2
        draw_outlined_text(draw, (l_t_x, tick_y_top + tick_l + 5), t_str, match_font, 
                           TITLE_COLOR if is_neutral else FOOTER_COLOR, stroke_width=1)
        l_x += l_sw

    # 5. Footer Signature
    footer_sig_y = HEIGHT - 50
    draw.line([(30, HEIGHT - 75), (WIDTH - 30, HEIGHT - 75)], fill=(40, 45, 50, 255), width=1)
    draw_outlined_text(draw, (30, footer_sig_y), "Data Source: Deadlock Community API", footer_font, FOOTER_COLOR, stroke_width=1)
    
    sig_text = "Analysis by @eReys13"
    sig_x = WIDTH - 30 - (footer_font.getbbox(sig_text)[2] - footer_font.getbbox(sig_text)[0])
    draw_outlined_text(draw, (sig_x, footer_sig_y), sig_text, footer_font, FOOTER_COLOR, stroke_width=1)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(output_file)
    print(f"Generated visual for {cat_info['name']} -> {output_file}")

if __name__ == "__main__":
    print(f"Reading CSV data from: {DATA_DIR}")
    print(f"Saving graphs to:     {OUTPUT_DIR}")
    for cat in CATEGORIES:
        generate_category_chart(cat)