# 📰 Fake News Detector

A machine learning web application that detects whether a news headline or article is **real** or **fake**. This project uses natural language processing (NLP) techniques and classification algorithms to analyze news content and classify it accurately.

---

## 🚀 Features

- Classifies news as **Real** or **Fake**
- Built with Python and scikit-learn
- Uses TF-IDF vectorization for text processing
- Simple and user-friendly web interface (Flask)
- Clean dataset preprocessing and model training

---


## 🧠 How It Works

1. The text input is vectorized using **TF-IDF**.
2. A trained model  Naive Bayes classifies it.
3. The result is displayed as either **Fake** or **Not Fake**.

---

## ▶️ Run the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py

