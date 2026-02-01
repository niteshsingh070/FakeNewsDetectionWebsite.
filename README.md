# 🕵️‍♂️ Fake News Detection System

A Machine Learning powered web application that detects whether a news article is **Real** or **Fake**.  
Trained on a massive dataset of over **100,000+ articles** (including 2024-2025 Indian News & AI-generated fake news) using **LinearSVC** and **TF-IDF Vectorization** to achieve high accuracy.

## 🚀 Live Demo
**[Click Here to Try the Web App](https://fakenewsdetectionwebsite-zziisglrrucvprf4axe9fq.streamlit.app/)** 
![User Interface](Screenshot 2026-02-01 235539.png)
![Confusion Matrix](confusion_matrix.png)
---

## 📌 Features
* **Real-time Detection:** Instantly classifies news articles as "Real" or "Fake".
* **Context Aware:** Specifically trained to handle **Indian Context**, **Technology**, and **Modern Geopolitics** (2025 data).
* **High Accuracy:** Uses a Linear Support Vector Classifier (LinearSVC) which outperforms standard Naive Bayes models for text classification.
* **Interactive UI:** Built with **Streamlit** for a clean, responsive experience (Dark Mode supported).

---

## 📊 Model Performance & Representation

The model is evaluated on a 25% test split. Below are the actual visualizations generated during the latest training session.

### 1. Confusion Matrix
This matrix shows the number of correct vs. incorrect predictions.
* **True Positives (Real):** Correctly predicted Real news.
* **True Negatives (Fake):** Correctly predicted Fake news.

![Confusion Matrix](confusion_matrix.png)

*(If the image is not loading, make sure `confusion_matrix.png` is uploaded to your GitHub repository)*

### 2. Performance Metrics
A graphical representation of Accuracy, Precision, Recall, and F1-Score.

![Performance Metrics](bar_chart.png)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **~98%** |
| **Precision** | **~99%** |
| **Recall** | **~99%** |
| **F1 Score** | **~99%** |

---

## 🛠️ Tech Stack
* **Language:** Python 3.12+
* **Machine Learning:** Scikit-Learn (LinearSVC, TF-IDF)
* **Data Processing:** Pandas, NumPy, Regular Expressions (Re)
* **Web Framework:** Streamlit
* **Serialization:** Joblib

---

## 📂 Project Structure
```bash
FAKE-NEWS-DETECTOR/
│
├── app.py                   # The Main Website (Streamlit App)
├── train_model.py           # ML Training Script (Generates .pkl files)
├── merge_data.py            # Data Cleaning & Merging Script
├── requirements.txt         # List of dependencies for cloud deployment
├── README.md                # Project Documentation
├── .gitignore               # Files to ignore (CSVs, large files)
│
├── models/                  # Saved ML Models (Auto-generated)
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── data/                    # Raw Data (Not uploaded to Git)
│   ├── custom_2025.csv      # Custom modern datasets
│   ├── gen_ai.csv           # AI-generated fake news
│   └── final_master_dataset.csv
│
└── images/                  # Visualizations
    ├── confusion_matrix.png
    └── bar_chart.png
