import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

warnings.filterwarnings('ignore')

def run(df):
    print("\n--- [4/9] EDA & Visualization (Generating 7 Plots) ---")
    
    # Ensure numerical stats are saved
    num_stats = df[['view_count', 'like_count', 'comment_count', 'duration']].describe().round(2)
    stats_path = os.path.join(OUTPUT_DIR, 'numerical_stats.csv')
    num_stats.to_csv(stats_path)
    
    # Create an EDA output directory
    EDA_DIR = os.path.join(OUTPUT_DIR, 'eda_plots')
    os.makedirs(EDA_DIR, exist_ok=True)
    
    # Set plot aesthetics
    sns.set_theme(style='whitegrid', palette='muted')
    
    # ---------------------------------------------------------
    # PLOT 1: Top 10 Video Languages
    # ---------------------------------------------------------
    print("1. Generating Language Distribution Plot...")
    if 'default_language' in df.columns:
        plt.figure(figsize=(10, 6))
        lang_counts = df['default_language'].fillna('Unknown').value_counts().head(10)
        sns.barplot(x=lang_counts.values, y=lang_counts.index, palette='viridis', hue=lang_counts.index, legend=False)
        plt.title('Top 10 Video Languages', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Videos')
        plt.ylabel('Language Code')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '01_language_distribution.png'), dpi=150)
        plt.close()
        
    # ---------------------------------------------------------
    # PLOT 2: Log Distribution of Engagement Metrics
    # ---------------------------------------------------------
    print("2. Generating Engagement Log Distribution Plot...")
    plt.figure(figsize=(12, 5))
    sns.histplot(np.log1p(df['view_count'].dropna()), bins=50, color='blue', alpha=0.5, label='Log Views', kde=True)
    sns.histplot(np.log1p(df['like_count'].dropna()), bins=50, color='green', alpha=0.5, label='Log Likes', kde=True)
    plt.title('Distribution of Views and Likes (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Log(Count + 1)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '02_engagement_distribution.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Views vs Likes Scatter Plot
    # ---------------------------------------------------------
    print("3. Generating Views vs Likes Scatter Plot...")
    plt.figure(figsize=(10, 6))
    # Subsample for scatter plot if dataset is massive to prevent freezing
    sample_df = df.sample(min(10000, len(df)), random_state=42) if len(df) > 10000 else df
    sns.scatterplot(x='view_count', y='like_count', data=sample_df, alpha=0.4, color='purple', s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Views vs. Likes Correlation (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Views (Log Scale)')
    plt.ylabel('Likes (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '03_views_vs_likes.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 4: Correlation Heatmap
    # ---------------------------------------------------------
    print("4. Generating Correlation Heatmap...")
    plt.figure(figsize=(8, 6))
    corr_cols = ['view_count', 'like_count', 'comment_count', 'duration']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '04_correlation_heatmap.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 5: Video Duration Distribution
    # ---------------------------------------------------------
    print("5. Generating Video Duration Plot...")
    plt.figure(figsize=(10, 6))
    # Filter out insanely long videos (e.g. 24hr live streams) for better viz
    duration_filtered = df[df['duration'] < 3600] # less than 1 hour
    sns.histplot(duration_filtered['duration'], bins=60, color='coral', kde=True)
    plt.title('Distribution of Video Durations (Under 1 Hour)', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (Seconds)')
    plt.ylabel('Count of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, '05_duration_distribution.png'), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 6: Uploads by Hour of Day
    # ---------------------------------------------------------
    print("6. Generating Uploads by Hour Plot...")
    if 'upload_hour' in df.columns:
        plt.figure(figsize=(10, 6))
        hour_counts = df['upload_hour'].value_counts().sort_index()
        sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker='o', color='teal', linewidth=2.5)
        plt.title('Upload Frequency by Hour of Day', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day (0-23)')
        plt.ylabel('Number of Uploads')
        plt.xticks(range(0, 24))
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '06_uploads_by_hour.png'), dpi=150)
        plt.close()

    # ---------------------------------------------------------
    # PLOT 7: Uploads by Day of Week
    # ---------------------------------------------------------
    print("7. Generating Uploads by Day Plot...")
    if 'upload_day_of_week' in df.columns:
        plt.figure(figsize=(10, 6))
        day_mapping = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        day_counts = df['upload_day_of_week'].value_counts().sort_index()
        day_counts.index = day_counts.index.map(day_mapping)
        sns.barplot(x=day_counts.index, y=day_counts.values, palette='magma', hue=day_counts.index, legend=False)
        plt.title('Upload Frequency by Day of the Week', fontsize=14, fontweight='bold')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Uploads')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, '07_uploads_by_day.png'), dpi=150)
        plt.close()

    print(f"All 7 plots successfully saved in {EDA_DIR}")
    return df

if __name__ == "__main__":
    from config import CLEAN_DATA_PATH
    df = pd.read_pickle(CLEAN_DATA_PATH)
    run(df)
