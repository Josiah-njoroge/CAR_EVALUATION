import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

#------------------------------------------------------
#  A TRANSFORMATION PIPELINE FOR PREPROCESSING THE DATA
#------------------------------------------------------ 

categorical_features =['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
def preprocessing_pipeline(categorical_features) :
    Pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]) 
    
    full_pipeline = ColumnTransformer(transformers=[
        ('cat', Pipeline, categorical_features)
    ])
    
    return full_pipeline
#-------------------------------------- 
def preprocess_data(input_filepath = RAW_DATA_DIR, output_filepath = PROCESSED_DATA_DIR):
    # Load raw data
    X = pd.read_csv(input_filepath / "X.csv")
    y = pd.read_csv(input_filepath / "y.csv")
    
    # Separate features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    # Create preprocessing pipeline
    pipeline = preprocessing_pipeline(categorical_features)  # Exclude target variable from features
    
    # Fit and transform the data
    X_processed = pipeline.fit_transform(X)
    
    # Convert processed data back to DataFrame
    processed_df = pd.DataFrame(X_processed.toarray())
    processed_df['class'] = y.values  # Add target variable back
    
    # Save processed data
    processed_df.to_csv(output_filepath, index=False)