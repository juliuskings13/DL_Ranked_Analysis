import json
import os
from pathlib import Path
import requests

# --- 1. PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
ICONS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.deadlock-api.com/v1/assets/icons"
HEADERS = {"User-Agent": "Mozilla/5.0 (Deadlock Analytics Dev)"}
SCHEMA_FILE = SCRIPT_DIR/ "icons_schema_results.json"

# --- 2. HELPER FUNCTIONS ---
def flatten_json(nested_json, current_path=""):
    """Flattens the nested asset JSON mapping into a clean path:url dict."""
    flat_map = {}
    for key, value in nested_json.items():
        if isinstance(value, dict):
            flat_map.update(flatten_json(value, f"{current_path}{key}/"))
        else:
            flat_map[f"{current_path}{key}"] = value
    return flat_map

# --- 3. CORE LOGIC ---
def download_all_assets():
    print(f"Fetching nested asset map from: {BASE_URL}")
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return

    icons_data = resp.json()

    # Save the master schema file into assets/icons/ for records
    with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(icons_data, f, indent=2)
    print(f"Master schema saved to: {SCHEMA_FILE}")

    # -------------------------------------------------------------
    # OPTION A: Download EVERYTHING (Heroes, Items, UI, etc.)
    # -------------------------------------------------------------
    # print("\nFlattening the entire asset map...")
    # icon_map = flatten_json(icons_data)

    # -------------------------------------------------------------
    # OPTION B: Download ONLY Hero Assets (What the code currently does)
    # -------------------------------------------------------------
    print("\nFlattening hero asset map...")
    heroes_subtree = icons_data.get("heroes", {})
    icon_map = flatten_json(heroes_subtree, current_path="heroes/")
    # -------------------------------------------------------------

    download_count = 0
    
    for relative_path, url in icon_map.items():
        # Target path mirrors the API structure (e.g., assets/icons/heroes/guns/shotgun.svg)
        destination_path = ICONS_DIR / relative_path
        
        # Filter: Only download images, skipping empty/invalid entries
        if not isinstance(url, str) or not (url.endswith(".png") or url.endswith(".svg")):
            continue
            
        # Optional filter from your script: target specific naming conventions
        #if not ("square" in relative_path or "hud" in relative_path or "selection" in relative_path or "_" not in relative_path):
           # continue

        # Ensure nested subdirectories exist
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading: {relative_path}")
        try:
            img_data = requests.get(url, headers=HEADERS, timeout=5).content
            with open(destination_path, "wb") as handler:
                handler.write(img_data)
            download_count += 1
        except Exception as e:
            print(f"Failed downloading {relative_path}: {e}")

    print(f"\n🎉 Success! Downloaded {download_count} assets into {ICONS_DIR}")

if __name__ == "__main__":
    download_all_assets()