# 📦 Amazon Reviews — Sentiment Analysis (3-Class)

End-to-end project for analyzing and classifying **Amazon product reviews** into **negative (0)**, **neutral (1)**, and **positive (2)** sentiments using **deep learning (TextCNN)** and **Streamlit** for deployment.

---

## 🚀 Project Overview
This project focuses on:
- Performing **EDA** to understand review distribution, text characteristics, and word usage patterns.
- Building and deploying a **TextCNN model** (Keras 3) for sentiment classification.
- Providing an **interactive Streamlit app** for:
  - Single text prediction
  - Batch prediction from CSV

---

## 🎯 Objectives
1. Conduct Exploratory Data Analysis (EDA):
   - Label distribution
   - Text length statistics
   - Boxplot per score
   - WordCloud visualization
2. Build a supervised deep learning model for sentiment classification.
3. Deploy the model with an easy-to-use interface (Streamlit).
4. Support both single and batch (CSV) predictions.

---

## 📂 Repository Structure
.<br>
├── eda.py # EDA visualization in Streamlit<br>
├── deployment/<br>
│ └── prediction.py # Sentiment prediction app<br>
├── artifacts_textcnn_v1/ # Saved model & metadata<br>
│ ├── model.keras<br>
│ └── meta.json<br>
├── amazon_reviews.csv # Dataset (raw reviews with score, content)<br>
├── requirements.txt # Python dependencies<br>
├── README.md # Project documentation<br>
└── ...

---

## 📊 Exploratory Data Analysis
EDA includes:
- **Label distribution** → shows imbalance (class 1 dominates, followed by class 5).
- **Text length distribution** → most reviews under 100 words.
- **Boxplot by score** → similar distributions across scores, but low scores often longer reviews.
- **WordClouds** → overall and per-score frequent words.

---

## 🤖 Model
- **Architecture**: TextCNN (Keras 3, TensorFlow backend)  
- **Classes**: `negative (0)`, `neutral (1)`, `positive (2)`  
- **Artifacts**:
  - `model.keras` → trained model
  - `meta.json` → label mapping & metadata

---

## 🌐 Streamlit App
### Features:
- **Single Text Prediction**: input a review → get sentiment + probabilities.
- **Batch CSV Prediction**: upload CSV → detect text column automatically → predict sentiments for all rows → download results.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Libraries**:
  - Data: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`
  - Machine Learning: `tensorflow`, `keras`
  - Deployment: `streamlit`
- **Deployment**: Streamlit (local/cloud)

---

## 📈 Example Output
- **EDA**: Label distribution, text length histograms, WordClouds
- **Prediction**:
  ```csv
  text, sentiment, prob_negatif, prob_netral, prob_positif
  "the product was amazing!", positive, 0.01, 0.03, 0.96
  "it’s okay, nothing special", neutral, 0.12, 0.78, 0.10
  "really bad quality, disappointed", negative, 0.89, 0.08, 0.03
