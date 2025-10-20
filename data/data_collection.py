from ucimlrepo import fetch_ucirepo 
import os
import pandas as pd
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_DIR

"""
    Fetches a dataset from the UCI Machine Learning Repository.

    Parameters:
    id (int): The unique identifier of the dataset to fetch.
    as_frame (bool): If True, returns the data as pandas DataFrames. Default is True.

    Returns:
    tuple: A tuple containing the features and targets of the dataset.
    """
def get_data(dataset_id: int, as_frame: bool = True, cache_dir = RAW_DATA_DIR):
    
    os.makedirs(cache_dir, exist_ok=True)
    X_path = os.path.join(cache_dir, "X.csv")
    y_path = os.path.join(cache_dir, "y.csv")
    
    # If data already exists, load it from disk
    if os.path.exists(X_path) and os.path.exists(y_path):
        print("☑ Loading data from disk.....")
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
        return X, y
    else:
        print("☑ Fetching data from UCI Repository.....")
        
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features 
        y = dataset.data.targets 
        metadata = dataset.metadata
        variables = dataset.variables
        
        # Save the data to disk for future use
        X.to_csv(X_path, index=False)
        y.to_csv(y_path, index=False)
    
    return X, y, metadata, variables

if __name__ == "__main__":
    X, y, metadata, variables = get_data(dataset_id=19)
    print("Features:\n", X.head())
    print("Targets:\n", y.head())
    print("Metadata:\n", metadata)
    print("Variables:\n", variables)
    
    """
Add versioning → Save with timestamps (e.g., X_2025_10_05.csv)

Add a YAML config → Store dataset_id, cache paths, and fetch options

Add exception handling → Handle API/network errors gracefully

Add logging instead of print statements → e.g., logging.info("✅ ...")
"""