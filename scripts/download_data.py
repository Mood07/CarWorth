"""
Run: python scripts/download_data.py
Requires: kaggle CLI configured (~/.kaggle/kaggle.json)
"""
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "austinreese/craigslist-carstrucks-data"

print(f"Downloading {DATASET} to {RAW_DIR} ...")
result = subprocess.run(
    ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
    capture_output=False
)
if result.returncode == 0:
    print("Download complete. File: data/raw/vehicles.csv")
else:
    print("Download failed. Make sure kaggle.json is in ~/.kaggle/")
    sys.exit(1)
