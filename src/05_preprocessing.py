import pandas as pd
import re
import unicodedata
from langdetect import detect
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, CLEAN_DATA_PATH, STOPWORDS

def detect_language(text):
    if not text or len(text.strip()) < 5:
        return 'Unknown'
    try:
        return detect(text[:200])
    except:
        return 'en'

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)

def run(df):
    print("\n--- [5/9] Multilingual Preprocessing Pipeline ---")
    
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df['text_combined'] = df['title'] + " " + df['description']
    
    print("Detecting languages...")
    if 'default_language' in df.columns:
        df['detected_lang'] = df['default_language'].fillna('en')
    else:
        df['detected_lang'] = df['text_combined'].apply(detect_language)

    print("Cleaning text and tokenizing...")
    df['cleaned_text'] = df['text_combined'].apply(clean_text)
    df['tokens'] = df['cleaned_text'].str.split()

    df = df[df['cleaned_text'].str.strip().str.len() > 2].reset_index(drop=True)
    
    print("Saving cleaned dataset to Pickle and CSV...")
    from config import CLEAN_CSV_PATH
    df.to_pickle(CLEAN_DATA_PATH)
    # Exclude columns containing lists/arrays if we want a clean CSV, or just save as is
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_DATA_PATH} and {CLEAN_CSV_PATH}")

    return df

if __name__ == "__main__":
    pass
