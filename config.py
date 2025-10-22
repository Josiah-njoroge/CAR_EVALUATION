from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SOURCE_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
