from ucimlrepo import fetch_ucirepo
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_DIR


def get_data(dataset_id: int, as_frame: bool = True, cache_dir=RAW_DATA_DIR):
    """
    Fetches a dataset from the UCI Machine Learning Repository and caches it locally.

    Parameters:
        dataset_id (int): The unique identifier of the dataset to fetch.
        as_frame (bool): If True, returns the data as pandas DataFrames. Default is True.
        cache_dir (str | Path): Directory where raw data will be stored.

    Returns:
        tuple: (X, y, metadata, variables)
    """

    os.makedirs(cache_dir, exist_ok=True)
    X_path = os.path.join(cache_dir, "X.csv")
    y_path = os.path.join(cache_dir, "y.csv")

    # ✅ Always return 4 values for consistency
    if os.path.exists(X_path) and os.path.exists(y_path):
        print("☑ Loading data from disk.....")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        metadata, variables = None, None  # No metadata cached
    else:
        print("☑ Fetching data from UCI Repository.....")
        dataset = fetch_ucirepo(id=dataset_id, as_frame=as_frame)
        X = dataset.data.features
        y = dataset.data.targets
        metadata = dataset.metadata
        variables = dataset.variables

        # Save data for future runs
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)

    return X, y, metadata, variables


if __name__ == "__main__":
    X, y, metadata, variables = get_data(dataset_id=19)

    print("\n✅ Data successfully loaded!\n")
    print("Features sample:\n", X.head(), "\n")
    print("Targets sample:\n", y.head(), "\n")

    if metadata:
        print("Metadata:\n", metadata)
    if variables:
        print("Variables:\n", variables)

    
    """
Add versioning → Save with timestamps (e.g., X_2025_10_05.csv)

Add a YAML config → Store dataset_id, cache paths, and fetch options

Add exception handling → Handle API/network errors gracefully

Add logging instead of print statements → e.g., logging.info("✅ ...")
"""