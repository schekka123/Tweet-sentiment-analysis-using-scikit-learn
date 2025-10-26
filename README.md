# Sentiment Analysis of Self-Driving Car Tweets

> End-to-end NLP project comparing **Bag-of-Words (TF-IDF)**, **MPQA subjectivity lexicon**, a **Hybrid (BoW+MPQA)** approach, and **Dependency-Triple features** for classifying tweet sentiment about autonomous vehicles.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50)
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔎 Overview

This repository implements and evaluates four sentiment-classification strategies on tweet data about **self-driving cars**:

1. **BoW (TF-IDF)**: tokenize → POS-filter (**NOUN/ADJ/VERB**) → lemmatize → TF-IDF n-grams → Linear SVM.
2. **MPQA Lexicon Features**: aggregate positive/negative/neutral counts per tweet as dense features.
3. **Hybrid (BoW + MPQA)**: augment the text with polarity hints (e.g., appending `positive`/`negative` tokens) before TF-IDF.
4. **Dependency Triples**: parse tweets, extract `(head, relation, dependent)` tuples, and hash as features.

**Labels:** original 5-point (plus “not relevant”) annotations collapsed to **four classes**:
`-1 = not relevant`, `1 = negative`, `2 = neutral`, `3 = positive`.

---

## 🧰 Tech Stack

- Python 3.9+
- scikit-learn for vectorization & classic ML (LinearSVC / Naive Bayes)
- NLTK for tokenization, POS tagging, and lemmatization
- (Optional) spaCy for dependency parsing (or NLTK + JPype if using MaltParser)
- tqdm, pandas, numpy for utilities/IO

---

## 🗂️ Repository Structure

    .
    ├─ data/
    │  └─ raw/                     # Put the CSV here (self-driving car tweets)
    ├─ mpqa/
    │  └─ subjectivity_lexicon/    # MPQA word lists (pos/neg/neutral)
    ├─ src/
    │  ├─ data.py                  # loading, label mapping, cleaning
    │  ├─ features_bow.py          # POS-filter, lemmatize, TF-IDF
    │  ├─ features_mpqa.py         # MPQA counters + token hints
    │  ├─ features_dep.py          # dependency triples
    │  ├─ models.py                # train/eval loops (SVM, NB)
    │  ├─ baselines.py             # VADER, Naive Bayes baselines
    │  └─ utils.py                 # seeds, metrics, logging
    ├─ notebooks/
    │  └─ exploration.ipynb        # optional EDA
    ├─ README.md
    └─ requirements.txt

> **Data:** Download the “Self-Driving Car Sentiment” CSV (CrowdFlower/Data.World) and place it under `data/raw/`. Ensure your loader maps labels to `{-1, 1, 2, 3}`.

---

## 📦 Installation

Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate

    pip install -U pip
    pip install -r requirements.txt

**Suggested `requirements.txt`:**

    pandas
    numpy
    scikit-learn
    nltk
    tqdm
    matplotlib
    # Choose ONE parsing path below:
    spacy
    # and then: python -m spacy download en_core_web_sm
    # OR use MaltParser via NLTK (requires Java + JPype):
    jpype1

First run: download NLTK resources:

    python -c "import nltk; [nltk.download(p) for p in ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4']]"

---

## 🚀 Quickstart (Experiments)

Train and evaluate different experiments:

    # BoW (unigrams)
    python -m src.models --exp bow --ngrams 1 1

    # BoW (uni+bi)
    python -m src.models --exp bow --ngrams 1 2

    # BoW (uni+bi+tri)
    python -m src.models --exp bow --ngrams 1 3

    # MPQA-only features
    python -m src.models --exp mpqa

    # Hybrid (BoW + MPQA hints)
    python -m src.models --exp hybrid --ngrams 1 1

    # Dependency triples
    python -m src.models --exp dep

    # Baselines
    python -m src.bas
