# 📺 YouTube NLP Data Analysis & Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Dash](https://img.shields.io/badge/Dash-Plotly-brightgreen.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red.svg)

A professional, fully modular Natural Language Processing (NLP) pipeline designed to ingest, process, and analyze massive YouTube datasets. This project processes over **306,000 YouTube videos**, performing everything from raw API extraction and text cleaning to generating Word2Vec embeddings and evaluating GPU-accelerated Machine Learning models.

## 🚀 Key Features

* **End-to-End Modular Architecture:** 9 distinct scripts cleanly separating data ingestion, EDA, NLP preprocessing, embedding generation, dimensionality reduction, ML classification, and UI rendering.
* **Multilingual Text Preprocessing:** Automatically detects video languages using `langdetect` and normalizes messy YouTube descriptions, stripping URLs, emojis, and stopwords.
* **Advanced Text Embeddings:** Generates both `TF-IDF` and contextual `Word2Vec` embeddings on the 300k+ corpus.
* **Dimensionality Reduction:** Compresses high-dimensional embedding spaces using **PCA** and **t-SNE** to map the linguistic features of hundreds of thousands of videos into stunning 2D scatter plots.
* **GPU-Accelerated Classification:** Evaluates `XGBoost`, `Linear SVM`, and `Logistic Regression` models to classify YouTube content formats, achieving **~82.5% F1-score** with XGBoost on TF-IDF.
* **Interactive Plotly Dashboard:** A robust local web app to explore engagement metrics (Views vs. Likes) dynamically filtered by language.

---

## 📁 Repository Structure

```text
youtube-nlp-project/
├── data/                                  # Ignored by Git (Contains raw youtube_data1.xlsx)
├── output/                                
│   ├── eda_plots/                         # High-quality EDA & ML Confusion Matrices
│   ├── dataset_info.txt                   # Pandas DataFrame summaries
│   ├── output.txt                         # Detailed Pipeline execution logs
│   ├── youtube_pca.png                    # 2D PCA visual space
│   └── youtube_tsne.png                   # 2D t-SNE visual space
├── src/                                   
│   ├── 01_install_dependencies.py         # Automates environment setup
│   ├── 02_youtube_api_extraction.py       # Queries YouTube Data v3 API
│   ├── 03_data_loading.py                 # Ingestion
│   ├── 04_eda_visualization.py            # Generates 7 data insight plots
│   ├── 05_preprocessing.py                # Text normalization & tokenization
│   ├── 06_embeddings.py                   # TF-IDF & Word2Vec models
│   ├── 07_dimensionality_reduction.py     # PCA / t-SNE mapping
│   ├── 08_classification.py               # Evaluates ML classifiers
│   └── 09_interactive_dashboard.py        # Generates the web app
├── dash_app.py                            # The interactive dashboard server
├── requirements.txt                       # Dependencies
└── run_pipeline.py                        # Master Orchestrator Script
```

---

## 📊 Visualizations & Insights

During execution, the pipeline automatically generates rich visualizations stored in `output/eda_plots/`:
1. **Engagement Metrics:** Correlation heatmaps and Log-scaled View vs. Like distributions.
2. **Temporal Analysis:** Upload frequencies mapped by Hour of the Day and Day of the Week.
3. **ML Performance:** Bar charts comparing Classifier Accuracy, Precision, Recall, and F1 scores across multiple embedding strategies.
4. **Confusion Matrices:** Heatmaps revealing exactly which linguistic categories the XGBoost model excels at identifying.

---

## 💻 How to Run

### 1. Run the Complete Pipeline
Ensure you have placed your raw dataset inside the `data/` folder as `youtube_data1.xlsx`.

```bash
# Execute the orchestrator
python run_pipeline.py
```
*Note: The t-SNE dimensionality reduction on 300,000 points is computationally heavy. Ensure you have sufficient RAM.*

### 2. Launch the Interactive Dashboard
To explore the processed dataset via an interactive UI:

```bash
python dash_app.py
```
Open `http://127.0.0.1:8050/` in your browser.

---
*Created for the Social Media Data Analysis & YouTube DVA Assignment.*
