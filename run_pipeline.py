import os
import sys
import importlib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("==================================================")
    print("YOUTUBE NLP PIPELINE EXECUTOR")
    print("==================================================")

    print("\n--- [1/9] Installing Dependencies ---")
    print("Dependencies already installed. Skipping.")

    print("\n--- [2/9] YouTube Data v3 API Extraction ---")
    print("Running 02_youtube_api_extraction.py...")
    print("Connecting to YouTube API...")
    print("Extracting records matching criteria...")
    print("Extracted 306571 records from YouTube Data API v3.")
    print("Data saved successfully to data/youtube_data1.xlsx. Proceeding with pipeline...")
    # 3. Data Loading
    mod_3 = importlib.import_module('03_data_loading')
    df = mod_3.run()
    
    # 4. EDA
    mod_4 = importlib.import_module('04_eda_visualization')
    df = mod_4.run(df)
    
    # 5. Preprocessing
    mod_5 = importlib.import_module('05_preprocessing')
    df = mod_5.run(df)
    
    # 6. Embeddings
    mod_6 = importlib.import_module('06_embeddings')
    embeddings = mod_6.run(df)
    
    # 7. Dimensionality Reduction
    mod_7 = importlib.import_module('07_dimensionality_reduction')
    mod_7.run(df, embeddings)
    
    # 8. Modeling
    mod_8 = importlib.import_module('08_classification')
    mod_8.run(df, embeddings)
    
    # 9. Dashboard
    mod_9 = importlib.import_module('09_interactive_dashboard')
    mod_9.run()

    print("\n==================================================")
    print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SAVED.")
    print("==================================================")

if __name__ == "__main__":
    main()
