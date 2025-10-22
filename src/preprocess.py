import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import sys

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

#------------------------------------------------------
#  CATEGORICAL FEATURES (exclude target)
#------------------------------------------------------
categorical_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

#------------------------------------------------------
#  Build preprocessing pipeline
#------------------------------------------------------
def preprocessing_pipeline(categorical_features):
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    full_pipeline = ColumnTransformer(transformers=[
        ('cat', cat_pipeline, categorical_features)
    ])
    
    return full_pipeline

#------------------------------------------------------
#  Preprocess data
#------------------------------------------------------
def preprocess_data(input_filepath=RAW_DATA_DIR, output_filepath=PROCESSED_DATA_DIR):
    print("📥 Loading raw data...")

    input_filepath = Path(input_filepath)

    # If user passes a file, read it directly
    if input_filepath.is_file():
        X = pd.read_csv(input_filepath)
        y = X['class']
        X = X.drop(columns=['class'])
    else:
        # Expect X.csv and y.csv inside directory
        X = pd.read_csv(input_filepath / "X.csv")
        y_path = input_filepath / "y.csv"
        if y_path.exists():
            y = pd.read_csv(y_path).squeeze()
        else:
            # Extract y from X if 'class' exists
            if 'class' in X.columns:
                y = X['class']
                X = X.drop(columns=['class'])
            else:
                raise FileNotFoundError("No 'y.csv' found and 'class' column missing in X.csv")

    print("⚙️ Building preprocessing pipeline...")
    pipeline = preprocessing_pipeline(categorical_features)

    print("🚀 Transforming data...")
    X_processed = pipeline.fit_transform(X)

    ohe_columns = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    processed_df = pd.DataFrame(X_processed.toarray(), columns=ohe_columns)
    processed_df['class'] = y.values

    output_filepath.mkdir(parents=True, exist_ok=True)
    output_file = output_filepath / "processed_data.csv"
    processed_df.to_csv(output_file, index=False)

    print(f"✅ Preprocessing complete! File saved to: {output_file}")
    return processed_df

#------------------------------
# SPLIT THE TRAINING DATA 
#------------------------------    
from sklearn.model_selection import train_test_split

def split_data(df, target_column='class', output_dir=PROCESSED_DATA_DIR, test_size=0.2, random_state=42):
    """
    Splits preprocessed data into train/test CSVs.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)
    
    print(f"✅ Train/test split complete! Files saved to {output_dir}")