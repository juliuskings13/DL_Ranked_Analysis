import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import pandas as pd 
from datetime import date
 
# --- CONFIGURATION ---
TITLE = "GLOBAL PRESENCE VS. WINRATE DIFFERENTIAL"
 
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
    
    # Fallback to auto-calculated week directory
    week_num = get_current_week_number()
    return Path.home() / "Projects" / "DL_ranked_analysis" / f"week{week_num}_analysis"
 
# Resolve directory
DATA_DIR = resolve_target_dir()
CSV_PATH = DATA_DIR / "global_presence_winrate.csv"
 
OUTPUT_DIR = DATA_DIR / "graphs"
OUTPUT_FILE = OUTPUT_DIR / "hero_presence_winrate_dumbbell_chart.png"
 
# --- THEME COLORS ---
BG_COLOR = (15, 23, 23, 255)          
TEXT_COLOR = (244, 240, 226, 255)     
TITLE_COLOR = (255, 208, 115, 255)    
FOOTER_COLOR = (140, 145, 155, 255)   
GRID_COLOR = (25, 35, 35, 255)        
 
# Refined Endpoint Node Colors (Separated from text/titles)
COLOR_WINRATE = (0, 235, 255, 255)     # Electric Cyan
COLOR_PRESENCE = (230, 83, 152, 255)   # Neon Orange/Coral
 
# Conditional Stem Status Colors
COLOR_STEM_GREEN = (65, 214, 126, 255) # High conversion (> 0.50)
COLOR_STEM_RED = (239, 99, 99, 255)    # Low conversion (< 0.50)
 
TEXT_STROKE_COLOR = (10, 15, 15, 255)
TEXT_STROKE_WIDTH = 2
 
# Dimensions
WIDTH = 1650
HEIGHT = 1000  # Extra height padding added to house the diagonal labels safely
PADDING = 40
 
# Visual Boundaries for the Vertical Bars
GRID_LEFT = 60
GRID_RIGHT = 1625
GRID_TOP = 200
GRID_BOTTOM = 700
GRID_WIDTH = GRID_RIGHT - GRID_LEFT
GRID_HEIGHT = GRID_BOTTOM - GRID_TOP
 
df = pd.read_csv(CSV_PATH)
 
df = df.rename(columns={
    "hero_name": "name",
    "global_winrate": "win",
    "global_presence": "pres"
    })
 
df = df.sort_values(by = "win", ascending= False)
 
 
DATA = df.to_dict(orient="records")
 
MIN_VAL = 0
MAX_VAL = 65
 
def get_centered_x(text, font, bounds):
    bbox = font.getbbox(str(text))
    return bounds[0] + ((bounds[1] - bounds[0]) - (bbox[2] - bbox[0])) // 2
 
def draw_outlined_text(draw, xy, text, font, fill, stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR):
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
 
def val_to_y(val):
    ratio = (val - MIN_VAL) / (MAX_VAL - MIN_VAL)
    return int(GRID_BOTTOM - (ratio * GRID_HEIGHT))
 
def generate_vertical_chart():
    image = Image.new("RGBA", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)
    
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        label_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        value_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        name_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        title_font = label_font = value_font = name_font = footer_font = ImageFont.load_default()
 
    # 1. Title
    t_x = get_centered_x(TITLE, title_font, (0, WIDTH))
    draw_outlined_text(draw, (t_x, PADDING), TITLE, title_font, TITLE_COLOR)
 
    # 2. Legend
    legend_items = [("Global Winrate", COLOR_WINRATE), ("Global Presence Rate", COLOR_PRESENCE)]
    leg_x = 650
    leg_y = PADDING + 60
    dot_radius = 7
    for label, color in legend_items:
        draw.ellipse([(leg_x, leg_y + 2), (leg_x + (dot_radius * 2), leg_y + 2 + (dot_radius * 2))], fill=color)
        draw_outlined_text(draw, (leg_x + (dot_radius * 2) + 10, leg_y), label, label_font, TEXT_COLOR)
        leg_x += 240
 
    # 3. Y Gridlines
    ticks = [0, 10, 20, 30, 40, 50, 60]
    for tick in ticks:
        y_pos = val_to_y(tick)
        draw.line([(GRID_LEFT, y_pos), (GRID_RIGHT, y_pos)], fill=GRID_COLOR, width=1)
        draw_outlined_text(draw, (GRID_LEFT - 55, y_pos - 10), f"{tick}%", label_font, FOOTER_COLOR, stroke_width=1)
 
    # 4. Render Columns
    col_count = len(DATA)
    col_spacing = GRID_WIDTH / (col_count - 1)
    node_radius = 7
    
    for idx, hero in enumerate(DATA):
        x_pos = int(GRID_LEFT + (idx * col_spacing))
        
        y_win = val_to_y(hero["win"])
        y_pres = val_to_y(hero["pres"])
        
        # Determine conversion logic: Winrate / Presence ratio
        conversion_ratio = hero["win"] / hero["pres"]
        stem_color = COLOR_STEM_GREEN if conversion_ratio > 0.50 else COLOR_STEM_RED
        
        # Stem Barbell Connector Line
        draw.line([(x_pos, y_win), (x_pos, y_pres)], fill=stem_color, width=4)
        
        # Value Circles
        draw.ellipse([(x_pos - node_radius, y_win - node_radius), (x_pos + node_radius, y_win + node_radius)], fill=COLOR_WINRATE)
        draw.ellipse([(x_pos - node_radius, y_pres - node_radius), (x_pos + node_radius, y_pres + node_radius)], fill=COLOR_PRESENCE)
        
        # Label Formatting Strings
        win_str = f"{hero['win']:.1f}%"
        pres_str = f"{hero['pres']:.1f}%"
        
        w_lbl_x = x_pos - (value_font.getbbox(win_str)[2] - value_font.getbbox(win_str)[0]) // 2
        p_lbl_x = x_pos - (value_font.getbbox(pres_str)[2] - value_font.getbbox(pres_str)[0]) // 2
        
        # Explicit placements: Presence ALWAYS above dot, Winrate ALWAYS below dot
        draw_outlined_text(draw, (p_lbl_x, y_pres - 22), pres_str, value_font, COLOR_PRESENCE, stroke_width=1)
        draw_outlined_text(draw, (w_lbl_x, y_win + 10), win_str, value_font, COLOR_WINRATE, stroke_width=1)
 
        # 5. Rotated Diagonal X-Axis Labels (Bottom-left to Top-right)
        name_str = hero["name"]
        n_bbox = name_font.getbbox(name_str)
        txt_w = n_bbox[2] - n_bbox[0]
        txt_h = n_bbox[3] - n_bbox[1]
        
        # Create clear scratchpad surface area big enough for rotation space
        pad_size = 140
        txt_img = Image.new("RGBA", (pad_size, pad_size), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        
        # Render text aligned to the right inside the scratchpad
        txt_draw.text((pad_size - txt_w - 5, pad_size // 2 - txt_h // 2), name_str, font=name_font, fill=TEXT_COLOR, 
                      stroke_width=TEXT_STROKE_WIDTH, stroke_fill=TEXT_STROKE_COLOR)
        
        # Rotate text image 45 degrees counter-clockwise to lift up to the right
        rotated_txt = txt_img.rotate(45, resample=Image.BICUBIC, expand=False)
        
        # Composite text directly onto the main frame axis coordinates
        image.alpha_composite(rotated_txt, (x_pos - pad_size + 15, GRID_BOTTOM + 5))
        draw.line([(x_pos, GRID_BOTTOM), (x_pos, GRID_BOTTOM + 8)], fill=(45, 55, 55, 255), width=1)
 
    # 6. Technical Insights Box Section
    note_box_y = GRID_BOTTOM + 130
    note_lines = [
        "ℹ️ Performance Metrics & Diagnostics:",
        "Connecting stems are color-coded relative to personal winrate.Stems are green if hero has >50% personal winrate, and red if it's <50%.",
        "Global presence is the percentage of all ranked games that hero is in. Global winrate is the percentage of all TOTAL ranked games won by said hero, regardless if they're in it or not."
    ]
    c_y = note_box_y
    for line in note_lines:
        c_color = TITLE_COLOR if "ℹ️" in line else FOOTER_COLOR
        draw_outlined_text(draw, (GRID_LEFT, c_y), line, footer_font, c_color, stroke_width=1)
        c_y += 24
 
    # 7. Signature Footer
    footer_y = HEIGHT - 40
    draw.line([(30, HEIGHT - 55), (WIDTH - 30, HEIGHT - 55)], fill=(40, 45, 50, 255), width=1)
    draw_outlined_text(draw, (30, footer_y), "Data Source: Deadlock Community API", footer_font, FOOTER_COLOR, stroke_width=1)
    
    sig_text = "Analysis by @eReys13"
    sig_x = WIDTH - 30 - (footer_font.getbbox(sig_text)[2] - footer_font.getbbox(sig_text)[0])
    draw_outlined_text(draw, (sig_x, footer_y), sig_text, footer_font, FOOTER_COLOR, stroke_width=1)
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_FILE)
    print(f"Successfully generated dynamic diagnostic dumbbell visual at: {OUTPUT_FILE}")
 
if __name__ == "__main__":
    generate_vertical_chart()
