import os
import sys
import csv
import math
from datetime import date
from pathlib import Path
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
CSV_FILE_PATH = DATA_DIR / "hero_type_diff_winrate.csv"
OUTPUT_FILE = OUTPUT_DIR / "team_composition_archetype_differentials.png"

TITLE = "HERO TYPE DIFFERENTIALS IN TEAM COMPOSITION"
SUBTITLE = "Impact of Class Dominance on Ranked Match Winrates"

# --- THEME COLORS ---
BG_COLOR = (15, 23, 23, 255)          
TEXT_COLOR = (244, 240, 226, 255)     
TITLE_COLOR = (255, 208, 115, 255)    
FOOTER_COLOR = (140, 145, 155, 255)   
GRID_COLOR = (25, 35, 35, 255)        
BASELINE_COLOR = (60, 70, 75, 255)    # 50% threshold line

# Core Data Series Colors
COLOR_GAIN = (65, 214, 126, 255)     # Green line/nodes (>50%)
COLOR_LOSS = (239, 99, 99, 255)      # Red line/nodes (<50%)
COLOR_TEAL = (30, 80, 80, 180)       # Semi-transparent Teal (Bar Color)

TEXT_STROKE_COLOR = (10, 15, 15, 255)
TEXT_STROKE_WIDTH = 2

# Canvas Dimensions
WIDTH = 1600
HEIGHT = 1300

# --- ADAPTIVE LAYOUT CONFIGURATION ---
MARGIN = 20  
PADDING_Y = 160
GAP = 30  
FOOTER_AREA_HEIGHT_PAD = 80 

TOTAL_GRID_W = WIDTH - (MARGIN * 2) - GAP
TOTAL_GRID_H = HEIGHT - PADDING_Y - FOOTER_AREA_HEIGHT_PAD - GAP

PANEL_W = TOTAL_GRID_W // 2
PANEL_H = TOTAL_GRID_H // 2

PANELS_POS = {
    "Brawlers":  (MARGIN, PADDING_Y),                             # Top-Left
    "Mystics":   (MARGIN + PANEL_W + GAP, PADDING_Y),             # Top-Right
    "Marksmen":  (MARGIN, PADDING_Y + PANEL_H + GAP),             # Bottom-Left
    "Assassins": (MARGIN + PANEL_W + GAP, PADDING_Y + PANEL_H + GAP)# Bottom-Right
}

# --- DATA LOADING ENGINE ---
def load_dataset_from_csv(filepath: Path) -> dict:
    """Reads hero_type_diff_winrate.csv dynamically into archetype dictionary lists."""
    dataset = {
        "Brawlers": [],
        "Mystics": [],
        "Marksmen": [],
        "Assassins": []
    }
    
    if not filepath.exists():
        print(f"Error: Target data file not found at {filepath}")
        sys.exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if not first_row:
            return dataset

        # Header detection check
        header_check = [cell.lower().strip() for cell in first_row]
        has_header = any(h in header_check for h in ['archetype', 'type', 'category', 'class', 'diff', 'win'])

        if has_header:
            arch_idx = next((i for i, h in enumerate(header_check) if h in ['archetype', 'type', 'category', 'class']), 0)
            diff_idx = next((i for i, h in enumerate(header_check) if 'diff' in h), 1)
            inst_idx = next((i for i, h in enumerate(header_check) if 'inst' in h or 'count' in h or 'match' in h), 2)
            win_idx  = next((i for i, h in enumerate(header_check) if 'win' in h or 'wr' in h), 3)
            rows = reader
        else:
            arch_idx, diff_idx, inst_idx, win_idx = 0, 1, 2, 3
            rows = [first_row] + list(reader)

        for row in rows:
            if not row or len(row) <= max(arch_idx, diff_idx, inst_idx, win_idx):
                continue

            raw_str = row[arch_idx].strip().lower()
            
            # Map robustly regardless of casing or singular/plural
            if "mark" in raw_str:
                archetype = "Marksmen"
            elif "brawl" in raw_str:
                archetype = "Brawlers"
            elif "myst" in raw_str:
                archetype = "Mystics"
            elif "assassin" in raw_str:
                archetype = "Assassins"
            else:
                archetype = raw_str.capitalize()

            if archetype in dataset:
                dataset[archetype].append({
                    "diff": int(float(row[diff_idx])),
                    "instances": int(float(row[inst_idx])),
                    "win": float(row[win_idx])
                })

    for arch in dataset:
        dataset[arch].sort(key=lambda x: x["diff"])
        
    return dataset

# --- HELPER FUNCTIONS ---
def get_centered_x(text, font, bounds):
    bbox = font.getbbox(str(text))
    return bounds[0] + ((bounds[1] - bounds[0]) - (bbox[2] - bbox[0])) // 2

def draw_outlined_text(draw, xy, text, font, fill, stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

def render_panel(image, label, data, position, fonts):
    draw = ImageDraw.Draw(image, "RGBA")
    p_x, p_y = position
    
    # Internal plot margins inside the panel box area
    graph_left = p_x + 60
    graph_right = p_x + PANEL_W - 20
    graph_top = p_y + 60
    graph_bottom = p_y + PANEL_H - 50
    
    g_w = graph_right - graph_left
    g_h = graph_bottom - graph_top

    # ALWAYS DRAW THE HEADER TITLE FOR THE PANEL
    draw_outlined_text(draw, (p_x + 10, p_y + 10), f" {label.upper()} DIFFERENTIAL", fonts['label'], TITLE_COLOR)

    # Core Boundaries Scaling Constants
    min_win, max_win = 40.0, 60.0  
    
    def win_to_y(w):
        clipped = max(min_win, min(max_win, w))
        ratio = (clipped - min_win) / (max_win - min_win)
        return int(graph_bottom - (ratio * g_h))

    # 1. Axis Tick Y Gridlines
    y_ticks = [40, 45, 50, 55, 60]
    for tick in y_ticks:
        y_val = win_to_y(tick)
        is_base = (tick == 50)
        line_w = 2 if is_base else 1
        line_c = BASELINE_COLOR if is_base else GRID_COLOR
        
        draw.line([(graph_left, y_val), (graph_right, y_val)], fill=line_c, width=line_w)
        draw_outlined_text(draw, (graph_left - 45, y_val - 8), f"{tick}%", fonts['value'], 
                           TITLE_COLOR if is_base else FOOTER_COLOR, stroke_width=1)

    # 2. X Axis Coordinates Calculation
    max_diff = max(item["diff"] for item in data) if data else 6
    max_diff = max(max_diff, 6)
    col_spacing = g_w / max_diff
    
    # Render X Ticks
    for d in range(max_diff + 1):
        x_val = int(graph_left + (d * col_spacing))
        draw.line([(x_val, graph_bottom), (x_val, graph_bottom + 6)], fill=GRID_COLOR, width=1)
        draw_outlined_text(draw, (x_val - 6, graph_bottom + 12), f"+{d}", fonts['value'], TEXT_COLOR, stroke_width=1)

    # If dataset has no records for this archetype, render notice & exit panel
    if not data:
        notice = "NO DATA AVAILABLE FOR THIS ARCHETYPE"
        n_x = get_centered_x(notice, fonts['label'], (graph_left, graph_right))
        draw_outlined_text(draw, (n_x, graph_top + g_h // 2), notice, fonts['label'], FOOTER_COLOR)
        return

    # 3. Sample Volume Bars
    max_sqrt = math.sqrt(100000)
    bar_width = int(col_spacing * 0.4)
    
    for item in data:
        d = item["diff"]
        inst = item["instances"]
        x_center = int(graph_left + (d * col_spacing))
        
        scaled_ratio = math.sqrt(inst) / max_sqrt
        bar_h = int(scaled_ratio * g_h)
        
        b_left = x_center - bar_width // 2
        b_top = graph_bottom - bar_h
        draw.rectangle([(b_left, b_top), (x_center + bar_width // 2, graph_bottom)], fill=COLOR_TEAL)
        
        if inst >= 1000:
            inst_str = f"{inst/1000:.1f}k"
        else:
            inst_str = str(inst)
        
        i_w = fonts['value'].getbbox(inst_str)[2] - fonts['value'].getbbox(inst_str)[0]
        draw_outlined_text(draw, (x_center - i_w // 2, graph_bottom - 18), inst_str, fonts['value'], 
                           (110, 130, 130, 200), stroke_width=1)

    # 4. Core Narrative Win Rate Line
    line_points = []
    for item in data:
        if item["instances"] <= 5: 
            continue
        
        d = item["diff"]
        x_pt = int(graph_left + (d * col_spacing))
        y_pt = win_to_y(item["win"])
        line_points.append((x_pt, y_pt))
        
    if len(line_points) > 1:
        draw.line(line_points, fill=TEXT_COLOR, width=3)

    # Node Overlay Dots
    node_r = 5
    for item in data:
        if item["instances"] <= 5: 
            continue
        d = item["diff"]
        win = item["win"]
        x_pt = int(graph_left + (d * col_spacing))
        y_pt = win_to_y(win)
        
        dot_color = COLOR_GAIN if win >= 50.0 else COLOR_LOSS
        draw.ellipse([(x_pt - node_r, y_pt - node_r), (x_pt + node_r, y_pt + node_r)], fill=dot_color)
        
        v_str = f"{win:.1f}%"
        v_w = fonts['value'].getbbox(v_str)[2] - fonts['value'].getbbox(v_str)[0]
        draw_outlined_text(draw, (x_pt - v_w // 2, y_pt - 20), v_str, fonts['value'], dot_color, stroke_width=1)

# --- MAIN GENERATION ENGINE ---
def generate_dashboard():
    dataset = load_dataset_from_csv(CSV_FILE_PATH)

    image = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    
    # Fonts
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        sub_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        value_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        title_font = sub_font = label_font = value_font = footer_font = ImageFont.load_default()

    fonts = {'label': label_font, 'value': value_font}

    # 1. Header
    t_x = get_centered_x(TITLE, title_font, (0, WIDTH))
    draw_outlined_text(draw, (t_x, 25), TITLE, title_font, TITLE_COLOR)
    
    s_x = get_centered_x(SUBTITLE, sub_font, (0, WIDTH))
    draw_outlined_text(draw, (s_x, 75), SUBTITLE, sub_font, FOOTER_COLOR)

    # 2. Render Panels (ALWAYS RENDER ALL PANELS)
    for name, pos in PANELS_POS.items():
        data = dataset.get(name, [])
        render_panel(image, name, data, pos, fonts)

    # 3. Footer
    footer_y = HEIGHT - 50
    draw.line([(30, HEIGHT - 65), (WIDTH - 30, HEIGHT - 65)], fill=(40, 45, 50, 255), width=1)
    
    legend_items = [
        {"text": "Win Rate Above 50% (Advantage)", "color": COLOR_GAIN, "type": "dot"},
        {"text": "Win Rate Below 50% (Liability)", "color": COLOR_LOSS, "type": "dot"},
        {"text": "Sample Instance Volumes (Muted Scaled Bars)", "color": COLOR_TEAL, "type": "bar"}
    ]
    
    current_leg_x = 30
    symbol_offset_y = footer_y + 2

    for item in legend_items:
        if item["type"] == "dot":
            node_r = 6
            draw.ellipse([(current_leg_x, symbol_offset_y - node_r), (current_leg_x + node_r*2, symbol_offset_y + node_r)], fill=item["color"])
            current_leg_x += node_r*2 + 8  
        elif item["type"] == "bar":
            bar_w = 14
            bar_h = 16
            draw.rectangle([(current_leg_x, symbol_offset_y - bar_h//2), (current_leg_x + bar_w, symbol_offset_y + bar_h//2)], fill=item["color"])
            current_leg_x += bar_w + 8  
            
        draw_outlined_text(draw, (current_leg_x, footer_y), item["text"], footer_font, FOOTER_COLOR, stroke_width=1)
        bbox = footer_font.getbbox(item["text"])
        txt_w = bbox[2] - bbox[0]
        current_leg_x += txt_w + 35  

    sig_text = "Analysis by @eReys13 • Source: Deadlock Community API"
    sig_x = WIDTH - 30 - (footer_font.getbbox(sig_text)[2] - footer_font.getbbox(sig_text)[0])
    draw_outlined_text(draw, (sig_x, footer_y), sig_text, footer_font, FOOTER_COLOR, stroke_width=1)

    # Save Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_FILE)
    print(f"Successfully rendered archetype dashboard to: {OUTPUT_FILE}")

if __name__ == "__main__":
    print(f"Reading CSV dataset from: {CSV_FILE_PATH}")
    print(f"Saving graph dashboard to: {OUTPUT_FILE}")
    generate_dashboard()