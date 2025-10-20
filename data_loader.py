from ucimlrepo import fetch_ucirepo
import os
import pandas as pd
import logging
import yaml
from datetime import datetime
from config import RAW_DATA_DIR, CONFIG_PATH

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/data_fetch.log", mode="a"),
        logging.StreamHandler()
    ]
)


# -----------------------------
# Load YAML Configuration
# -----------------------------
def load_yaml_config(config_path=CONFIG_PATH):
    """Loads the dataset configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# -----------------------------
# Fetch Dataset Function
# -----------------------------
def get_data(dataset_id: int, as_frame: bool = True, cache_dir=RAW_DATA_DIR):
    """
    Fetches a dataset from the UCI Machine Learning Repository.
    Caches it locally for re-use and includes timestamp versioning.

    Parameters:
        dataset_id (int): UCI dataset ID
        as_frame (bool): Return pandas DataFrame (default True)
        cache_dir (str): Directory for storing cached datasets
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Timestamped versioning
    timestamp = datetime.now().strftime("%Y_%m_%d")
    X_path = os.path.join(cache_dir, f"X_{timestamp}.csv")
    y_path = os.path.join(cache_dir, f"y_{timestamp}.csv")

    try:
        # Check for existing cached dataset
        existing_X = [f for f in os.listdir(cache_dir) if f.startswith("X_")]
        existing_y = [f for f in os.listdir(cache_dir) if f.startswith("y_")]

        if existing_X and existing_y:
            latest_X = os.path.join(cache_dir, sorted(existing_X)[-1])
            latest_y = os.path.join(cache_dir, sorted(existing_y)[-1])
            logging.info(f"☑ Loading cached data: {latest_X}, {latest_y}")
            X = pd.read_csv(latest_X)
            y = pd.read_csv(latest_y)
            return X, y, None, None

        # Fetch new data if not cached
        logging.info(f"☑ Fetching dataset ID {dataset_id} from UCI Repository...")
        dataset = fetch_ucirepo(id=dataset_id, as_frame=as_frame)

        X = dataset.data.features
        y = dataset.data.targets
        metadata = dataset.metadata
        variables = dataset.variables

        # Save fetched data with timestamp
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
        logging.info(f"✅ Dataset saved to {cache_dir} as versioned CSVs")

        return X, y, metadata, variables

    except Exception as e:
        logging.error(f"❌ Error while fetching dataset {dataset_id}: {e}")
        raise


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    try:
        cfg = load_yaml_config()
        dataset_id = cfg.get("dataset_id", 19)
        as_frame = cfg.get("as_frame", True)

        X, y, metadata, variables = get_data(dataset_id, as_frame)
        logging.info(f"✅ Features preview:\n{X.head()}")
        logging.info(f"✅ Targets preview:\n{y.head()}")
        logging.info(f"📄 Metadata:\n{metadata}")
        logging.info(f"📊 Variables:\n{variables}")

    except Exception as e:
        logging.error(f"❌ Program terminated unexpectedly: {e}")
