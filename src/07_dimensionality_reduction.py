import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, TSNE_SAMPLE_SIZE, RANDOM_STATE

def run(df, embeddings):
    print("\n--- [7/9] Dimensionality Reduction (PCA / t-SNE) ---")
    
    X_raw = embeddings.get('word2vec', embeddings['tfidf'])
    
    print(f"Using the ENTIRE dataset ({len(X_raw)} rows) for t-SNE. Warning: This may take a very long time!")
    df_plot = df.copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("Running PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_res = pca.fit_transform(X)

    # Set a highly distinct palette for the languages
    distinct_palette = sns.color_palette("bright", n_colors=6)

    # Plot PCA
    plt.figure(figsize=(12, 10))
    top_langs = df_plot['detected_lang'].value_counts().nlargest(5).index
    df_plot['plot_lang'] = df_plot['detected_lang'].apply(lambda x: x if x in top_langs else 'Other')
    
    sns.scatterplot(
        x=pca_res[:, 0], y=pca_res[:, 1], 
        hue=df_plot['plot_lang'], 
        palette=distinct_palette, 
        s=1.5,           # Much smaller dot size for 300k points
        alpha=0.25,      # High transparency to show density
        linewidth=0,     # No borders on dots
        legend='full'
    )
    plt.title('PCA of YouTube Text Embeddings (N=306,285)', fontsize=16, fontweight='bold')
    # Make legend points larger so they are visible
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Language", markerscale=10)
    plt.tight_layout()
    
    pca_path = os.path.join(OUTPUT_DIR, 'youtube_pca.png')
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')  # Higher DPI for crispness
    plt.close()
    print(f"PCA plot saved to {pca_path}")

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=RANDOM_STATE)
    tsne_res = tsne.fit_transform(X)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_res[:, 0], y=tsne_res[:, 1], 
        hue=df_plot['plot_lang'], 
        palette=distinct_palette, 
        s=1.5, 
        alpha=0.25, 
        linewidth=0,
        legend='full'
    )
    plt.title('t-SNE of YouTube Text Embeddings (N=306,285)', fontsize=16, fontweight='bold')
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Language", markerscale=10)
    plt.tight_layout()
    
    tsne_path = os.path.join(OUTPUT_DIR, 'youtube_tsne.png')
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {tsne_path}")

if __name__ == "__main__":
    pass
