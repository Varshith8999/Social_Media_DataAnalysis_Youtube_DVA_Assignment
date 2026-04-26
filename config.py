import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'youtube_data1.xlsx')
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, 'youtube_cleaned.pkl')
CLEAN_CSV_PATH = os.path.join(OUTPUT_DIR, 'youtube_cleaned.csv')
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'youtube_embeddings.pkl')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

RANDOM_STATE = 42
TSNE_SAMPLE_SIZE = 10000
MAX_TFIDF_FEATURES = 1000
W2V_VECTOR_SIZE = 128

STOPWORDS = {'the','a','an','is','in','it','of','and','to','for','this','that','are','was','with','on','as','by','at','or','from','be','you','your','we','our'}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
