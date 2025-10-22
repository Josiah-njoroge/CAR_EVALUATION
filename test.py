from config import PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR
import pandas as pd

df = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
print(df.head())

F = pd.read_csv(RAW_DATA_DIR / 'X.csv')
y = pd.read_csv(RAW_DATA_DIR / 'y.csv')

print(F.head())
print(y.head())
