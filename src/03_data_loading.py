import pandas as pd
import os
import sys

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, OUTPUT_DIR

def run():
    print("\n--- [3/9] Data Loading ---")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {RAW_DATA_PATH}")

    print(f"Loading entire dataset from {RAW_DATA_PATH}...")
    df = pd.read_excel(RAW_DATA_PATH)
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")

    info_path = os.path.join(OUTPUT_DIR, 'dataset_info.txt')
    df.info(buf=open(info_path, 'w'))
    print(f"Dataset info saved to {info_path}")

    return df

if __name__ == "__main__":
    run()
