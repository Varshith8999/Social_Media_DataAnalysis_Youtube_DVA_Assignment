import os
import sys
from config import DATA_DIR, OUTPUT_DIR
import importlib

def main():
    print("==================================================")
    print("STARTING MODULAR YOUTUBE NLP PIPELINE")
    print("==================================================")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Import modules dynamically to ensure clean namespace
    try:
        load_eda = importlib.import_module('01_load_and_eda')
        preprocess = importlib.import_module('02_preprocess')
        embeddings_mod = importlib.import_module('03_embeddings')
        visualization = importlib.import_module('04_visualization')
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

    # Execute pipeline sequentially passing the data in-memory
    df = load_eda.run()
    df = preprocess.run(df)
    embeddings = embeddings_mod.run(df)
    visualization.run(df, embeddings)

    print("\n==================================================")
    print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SAVED IN 'output/'")
    print("==================================================")

if __name__ == "__main__":
    main()
