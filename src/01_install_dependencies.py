import subprocess
import sys

def run():
    print("\n--- [1/9] Installing Dependencies ---")
    packages = [
        "pandas", "numpy", "matplotlib", "seaborn", "wordcloud", 
        "langdetect", "scikit-learn", "gensim", "openpyxl",
        "google-api-python-client", "plotly", "dash"
    ]
    
    # We skip torch, transformers, and umap-learn here because of your local 
    # Application Control Policy blocking their DLLs.
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            
    print("Dependencies installation complete.")

if __name__ == "__main__":
    run()
