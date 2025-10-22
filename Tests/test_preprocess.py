import os
import sys
import pytest
import pandas as pd

# -------------------------------------------------------------------------
# Add the project root to the Python path so imports like "from src..." work
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import preprocessing functions
from src.preprocess_data import preprocess_data, split_data, preprocessing_pipeline


# -------------------------------------------------------------------------
# Fixture: Creates temporary sample data for testing
# -------------------------------------------------------------------------

@pytest.fixture
def sample_data(tmp_path):
    data = pd.DataFrame({
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more', '2'],
        'lug_boot': ['small', 'med', 'big', 'small'],
        'safety': ['low', 'med', 'high', 'low'],
        'class': ['unacc', 'acc', 'good', 'vgood']
    })
    X = data.drop(columns=['class'])
    y = data[['class']]
    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False)
    return tmp_path

# -------------------------------------------------------------------------
# Test 1: Preprocessing works and output file is created
# -------------------------------------------------------------------------
def test_preprocess_data_creates_encoded_output(sample_data):
    """Test if preprocess_data correctly encodes and saves processed file"""
    input_dir, output_dir = sample_data

    preprocess_data(input_filepath=input_dir, output_filepath=output_dir)

    processed_file = output_dir / "processed_data.csv"
    assert processed_file.exists(), "Processed data file was not created."

    df = pd.read_csv(processed_file)
    assert "class" in df.columns, "Target column 'class' is missing."
    assert not df.empty, "Processed file is empty."


# -------------------------------------------------------------------------
# Test 2: Split function returns correct shapes
# -------------------------------------------------------------------------
def test_split_data_returns_correct_shapes(sample_data):
    """Test split_data function for correct train/test splits"""
    input_dir, output_dir = sample_data

    preprocess_data(input_filepath=input_dir, output_filepath=output_dir)
    processed_file = output_dir / "processed_data.csv"

    df = pd.read_csv(processed_file)
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    assert len(X_train) + len(X_test) == len(X), "Split sizes don't add up."
    assert len(y_train) + len(y_test) == len(y), "Target splits don't match feature splits."
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions differ between splits."
