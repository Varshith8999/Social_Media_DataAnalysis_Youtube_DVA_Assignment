import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLEAN_DATA_PATH, EMBEDDINGS_PATH, MAX_TFIDF_FEATURES, W2V_VECTOR_SIZE

def run(df):
    print("\n--- [6/9] Generating Embeddings ---")
    
    corpus = df['cleaned_text'].tolist()
    token_corpus = df['tokens'].tolist()
    embeddings = {}

    print("Generating TF-IDF...")
    tfidf = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES)
    embeddings['tfidf'] = tfidf.fit_transform(corpus).toarray()

    print("Generating Word2Vec...")
    w2v_model = Word2Vec(sentences=token_corpus, vector_size=W2V_VECTOR_SIZE, window=5, min_count=1, workers=4, epochs=5)
    def sentence_vector_w2v(tokens, model, size):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(size)
        
    embeddings['word2vec'] = np.array([sentence_vector_w2v(tok, w2v_model, W2V_VECTOR_SIZE) for tok in token_corpus])
    
    # We skip FastText, mBERT, and LaBSE due to Windows Application Control Policy on DLLs
    print("Skipped FastText, mBERT, LaBSE due to missing dependencies.")

    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

if __name__ == "__main__":
    pass
