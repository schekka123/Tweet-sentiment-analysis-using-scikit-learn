# Sentiment Analysis of Self-Driving Car Tweets

> End-to-end NLP project comparing **Bag-of-Words (TF-IDF)**, **MPQA subjectivity lexicon**, a **Hybrid (BoW+MPQA)** approach, and **Dependency-Triple features** for classifying tweet sentiment about autonomous vehicles.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50)
![spaCy](https://img.shields.io/badge/spaCy-Parser-09A3D5) <!-- Or MaltParser -->
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey) <!-- Placeholder License -->

---

## 🔎 Overview

This repository implements and evaluates four sentiment-classification strategies on tweet data about **self-driving cars**:

1.  **BoW (TF-IDF)**: tokenize → POS-filter (**NOUN/ADJ/VERB**) → lemmatize → TF-IDF n-grams → Linear SVM.
2.  **MPQA Lexicon Features**: aggregate positive/negative/neutral counts per tweet as dense features.
3.  **Hybrid (BoW + MPQA)**: augment the text with polarity hints (e.g., appending `positive`/`negative` tokens) before TF-IDF.
4.  **Dependency Triples**: parse tweets, extract `(head, relation, dependent)` tuples, and hash/vectorize as features.

**Labels:** original 5-point (plus “not relevant”) annotations collapsed to **four classes**: `-1 = not relevant`, `1 = negative`, `2 = neutral`, `3 = positive`.

---

## 🧰 Tech Stack

- Python 3.9+
- `scikit-learn` for vectorization & classic ML (LinearSVC / Naive Bayes)
- `NLTK` for tokenization, POS tagging, and lemmatization (and potentially MaltParser interface)
- `spaCy` (Optional) for dependency parsing
- `vaderSentiment` for baseline comparison
- `pandas`, `numpy` for data handling
- `matplotlib` for plotting
- `tqdm` for progress bars

---

## 📊 Dataset & Label Mapping

- **Dataset:** CrowdFlower/Data.World “[Sentiment self-driving cars](https://data.world/crowdflower/sentiment-self-driving-cars)” (~**7,156** tweets).
- **Original labels:** `1` (Very Negative) → `5` (Very Positive) + `-1` (*Not Relevant*).
- **Collapsed labels used here:**
    - `-1` → **Not Relevant** (Original -1)
    - `1` → **Negative** (Merged Original 1 & 2)
    - `2` → **Neutral** (Original 3)
    - `3` → **Positive** (Merged Original 4 & 5)

*Why collapse?* Reduces sparsity/noise in extreme classes and stabilizes training, especially given the dataset size.

---

## 📦 Installation

Create and activate a virtual environment, then install dependencies:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# 2. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install required packages
pip install -r requirements.txt
# (Ensure requirements.txt includes: scikit-learn nltk pandas numpy matplotlib tqdm vaderSentiment spacy)
# Or manually: pip install scikit-learn nltk pandas numpy matplotlib tqdm vaderSentiment spacy

# 4. Download necessary NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Download spaCy model (if using spaCy for parsing)
python -m spacy download en_core_web_sm

# 6. Download the dataset CSV from data.world
#    Place 'sentiment_self_driving_cars.csv' (or similar) into a `data/` subdirectory.

# 7. (Optional) Setup MaltParser if using NLTK's interface instead of spaCy
#    Follow MaltParser installation and NLTK integration instructions.
