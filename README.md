# Sentiment Analysis of Self-Driving Car Tweets

> End-to-end NLP project comparing **Bag-of-Words (TF-IDF)**, **MPQA subjectivity lexicon**, a **Hybrid (BoW+MPQA)** approach, and **Dependency-Triple features** for classifying tweet sentiment about autonomous vehicles.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50)
![spaCy](https://img.shields.io/badge/spaCy-Parser-09A3D5)
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔎 Overview

This repository implements and evaluates four sentiment-classification strategies on tweet data about **self-driving cars**:

1. **BoW (TF-IDF)**: tokenize → POS-filter (**NOUN/ADJ/VERB**) → lemmatize → TF-IDF n-grams → Linear SVM.  
2. **MPQA Lexicon Features**: aggregate positive/negative/neutral counts per tweet as dense features.  
3. **Hybrid (BoW + MPQA)**: augment the text with polarity hints (e.g., appending `positive`/`negative` tokens) before TF-IDF.  
4. **Dependency Triples**: parse tweets, extract `(head, relation, dependent)` tuples, and hash as features.

**Labels:** original 5-point (plus “not relevant”) annotations collapsed to **four classes**: `-1 = not relevant`, `1 = negative`, `2 = neutral`, `3 = positive`.

---

## 🧰 Tech Stack

- Python 3.9+  
- scikit-learn for vectorization & classic ML (LinearSVC / Naive Bayes)  
- NLTK for tokenization, POS tagging, and lemmatization  
- (Optional) spaCy for dependency parsing (or NLTK + JPype if using MaltParser)  
- tqdm, pandas, numpy, matplotlib for utilities/IO/plots

---

## 📊 Dataset & Label Mapping

- **Dataset:** CrowdFlower/Data.World “Self-Driving Car Sentiment” (~**7,156** tweets).  
- **Original labels:** `1–5` (1 very negative → 5 very positive) + `-1` (*not relevant*).  
- **Collapsed labels used here:**  
  - `-1` → *not relevant*  
  - `1`  → *negative* (merge original 1 & 2)  
  - `2`  → *neutral* (original 3)  
  - `3`  → *positive* (merge original 4 & 5)  

*Why collapse?* Reduces sparsity/noise in extreme classes and stabilizes training on tweets.

---

## 📦 Installation

Create and activate a virtual environment, then install dependencies:
