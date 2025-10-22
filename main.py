#------------------------------------------------------
#  Main function to run script
#------------------------------------------------------
from src.preprocess import preprocess_data, split_data


def main():
    print("🏗️ Starting preprocessing pipeline...")

    # Step 1: Preprocess data
    processed_df = preprocess_data()

    # Step 2: Split data into train/test sets
    split_data(processed_df)

    print("🎉 All done! Preprocessed and split data is ready for ML.")
    
if __name__ == "__main__":
    main()