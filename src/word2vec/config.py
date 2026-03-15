from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "saved_models"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define standard experiment sizes (number of tokens to read)
DATASET_SIZES = {
    "small": 1_000_000,     # For fast testing/reviewing (seconds)
    "medium": 5_000_000,  # Good balance for CPU training (it took few seconds on my MacBook Pro)
    "big": None           # Reads the entire 17M token dataset (Will take a long time on CPU!)
}