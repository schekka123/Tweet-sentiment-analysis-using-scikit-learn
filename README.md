```markdown
# Sentiment Analysis of Self-Driving Car Tweets

> End-to-end NLP project evaluating **Bag-of-Words (TF-IDF)**, **MPQA subjectivity lexicon**, a **Hybrid (BoW+MPQA)** approach, and **Dependency-Triple features** to classify tweet sentiment about autonomous vehicles.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4CAF50)
![spaCy](https://img.shields.io/badge/spaCy-Parser-09A3D5)
![Status](https://img.shields.io/badge/Status-Academic%20Project-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔍 Abstract

We build four custom sentiment models for self-driving car tweets: **(1)** TF-IDF Bag-of-Words (BoW), **(2)** **MPQA** lexicon-based polarity features, **(3)** **Hybrid** BoW augmented with MPQA polarity hints, and **(4)** **Dependency Triples** derived from a dependency parser. Using **NLTK** and **scikit-learn**, we compare accuracy and macro-F1 on the CrowdFlower dataset; performance spans roughly **0.55–0.65** accuracy.

---

## 📊 Dataset & Label Mapping

- **Source:** CrowdFlower/Data.World *Self-Driving Car Sentiment* (~**7,156** tweets).
- **Original labels:** `1–5` plus `-1` (*not relevant*).
- **Collapsed labels used:** `-1` (*not relevant*), `1` (*negative = 1–2*), `2` (*neutral = 3*), `3` (*positive = 4–5*).

**Why collapse?** Sparse extremes and labeling noise in tweets; the merge stabilizes training and evaluation.

---

## 🧠 Methods

### 1) Bag-of-Words (TF-IDF)
Tokenize → **POS-tag** → keep **NOUN/ADJ/VERB** → **lemmatize** (WordNet) → `TfidfVectorizer` → linear SVM.

### 2) MPQA Subjectivity Lexicon
Count **positive / negative / neutral** tokens per tweet (MPQA), vectorize with `DictVectorizer`, train SVM.

### 3) Hybrid (BoW + MPQA)
Append **polarity tokens** (repeat `positive`/`negative` based on counts) to the lemmatized text before TF-IDF to inject a prior.

### 4) Dependency Triples
Parse tweets (MaltParser via NLTK or spaCy) to extract `(head, relation, dependent)` triples; hash as features and train SVM.

---

## ✅ Results (Validation Accuracy)

| Model / Features                          | Accuracy |
|-------------------------------------------|---------:|
| **Hybrid (BoW + MPQA polarity tokens)**   | **0.648** |
| BoW (unigrams)                            | 0.637 |
| BoW (uni+bi)                              | 0.620 |
| BoW (uni+bi+tri)                          | 0.610 |
| Dependency Triples                        | 0.584 |
| VADER (baseline)                          | 0.569 |
| Naive Bayes (baseline)                    | 0.550 |
| MPQA-only (lexicon counts)                | 0.549 |

**Notes**
- **BoW** is robust on tweets; **Hybrid** adds a small but consistent bump.
- **More n-grams ≠ better** on short/noisy text; tri-grams underperform due to sparsity/noise.
- **Lexicon-only** features struggle with slang/misspellings; **dependency features** slightly beat NB.

> Report **macro-F1** alongside accuracy due to class imbalance.

---

## 🧪 Evaluation Protocol

- **Split:** Stratified train/validation (e.g., 80/20) or K-fold CV.
- **Metrics:** Accuracy (headline), **macro-F1** (recommended), per-class precision/recall/F1.
- **Artifacts:** Save trained model/vectorizer and logs for reproducibility.
- **Randomness:** Fix seeds across NumPy/Sklearn; document package versions.

---

## 🗂 Repository Structure

```

.
├─ data/
│  └─ raw/                     # CrowdFlower/Data.World CSV
├─ mpqa/
│  └─ subjectivity_lexicon/    # MPQA lists (pos/neg/neutral)
├─ src/
│  ├─ data.py                  # load/normalize, label mapping {-1,1,2,3}
│  ├─ features_bow.py          # POS-filter (NOUN/ADJ/VERB), lemmatize, TF-IDF
│  ├─ features_mpqa.py         # MPQA counts & polarity-token appending
│  ├─ features_dep.py          # dependency triples -> features
│  ├─ models.py                # training/eval (LinearSVC, NB) + CLI
│  ├─ baselines.py             # VADER & Naive Bayes
│  └─ utils.py                 # seeding, metrics, logging
├─ notebooks/
│  └─ exploration.ipynb        # optional EDA
├─ README.md
└─ requirements.txt

````

---

## ⚙️ Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
````

**Suggested `requirements.txt`:**

```
pandas
numpy
scikit-learn
nltk
tqdm
matplotlib
# Choose ONE parsing stack:
spacy
# then: python -m spacy download en_core_web_sm
# OR (NLTK + MaltParser alternative; requires Java):
jpype1
```

**First run (download NLTK resources):**

```
python -c "import nltk; [nltk.download(p) for p in ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4']]"
```

---

## 🚀 Quickstart (Experiments)

```
# BoW (unigrams)
python -m src.models --exp bow --ngrams 1 1

# BoW (uni+bi)
python -m src.models --exp bow --ngrams 1 2

# BoW (uni+bi+tri)
python -m src.models --exp bow --ngrams 1 3

# MPQA-only
python -m src.models --exp mpqa

# Hybrid (BoW + MPQA polarity tokens)
python -m src.models --exp hybrid --ngrams 1 1

# Dependency triples
python -m src.models --exp dep

# Baselines
python -m src.baselines --exp vader
python -m src.baselines --exp nb
```

Each run prints accuracy; pass `--report` to include macro-F1 and a concise classification report.

---

## 🔬 Implementation Details

**Text Normalization**

* Lowercase, strip URLs/user handles, normalize elongated words (optional).
* Preserve emojis/emoticons if you plan to map them to sentiment tokens.

**POS-Filtering & Lemmatization**

* Keep **NOUN/ADJ/VERB** using NLTK tagger; map to WordNet POS and lemmatize.

**Vectorization**

* `TfidfVectorizer` with tuned `min_df`, `max_df`, and `ngram_range`.
* Try `sublinear_tf=True` and cap `max_features` if needed.

**Classifiers**

* Primary: `LinearSVC` (fast, strong on sparse text).
* Baselines: Multinomial Naive Bayes; rule-based **VADER**.

**MPQA Features**

* Per-tweet positive/negative/neutral counts from the MPQA lexicon.
* Hybrid trick: append polarity tokens (repeat `positive` *k* times) before TF-IDF.

**Dependency Triples**

* Use **spaCy `en_core_web_sm`** (simpler) or **MaltParser** (via NLTK + Java).
* Hash triples `"head_REL_dep"` into sparse features; prune rare triples.

---

## 🧪 Reproducibility Checklist

* Set seeds (NumPy / sklearn).
* Save artifacts:

  * `artifacts/model.joblib`
  * `artifacts/vectorizer.joblib`
  * `artifacts/label_encoder.joblib` (if used)
* Log environment:

  * `python -m pip freeze > artifacts/requirements.lock`

---

## 🧰 CLI Examples

```
# Hybrid with unigrams and a detailed report
python -m src.models --exp hybrid --ngrams 1 1 --report

# Dependency triples with pruning and a different C
python -m src.models --exp dep --triple-min-count 3 --C 0.5

# Save artifacts and confusion matrix
python -m src.models --exp bow --ngrams 1 2 --save-artifacts --report
```

---

## 🧩 Error Analysis Tips

* Inspect top false positives/negatives to find systematic patterns (sarcasm, negation, hashtags).
* Check out-of-vocabulary rates vs. MPQA; slang and domain terms may need custom lists.
* Try emoji→polarity mapping and hashtag segmentation to recover additional signal.

---

## 📚 References

* Brownlee, J. (2019, August 7). *A Gentle Introduction to the Bag-of-Words Model*. Machine Learning Mastery. [https://machinelearningmastery.com/gentle-introduction-bag-words-model/](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
* CrowdFlower. (2016, November 21). *Sentiment – Self-Driving Cars* (dataset). data.world. [https://data.world/crowdflower/sentiment-self-driving-cars](https://data.world/crowdflower/sentiment-self-driving-cars)
* Hutto, C. J., & Gilbert, E. E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. Proc. ICWSM-14, Ann Arbor, MI.
* Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
* Nivre, J., Hall, J., & Nilsson, J. (2006). *MaltParser: A data-driven parser-generator for dependency parsing*. In *LREC* (Vol. 6, pp. 2216–2219).
* Wilson, T., Wiebe, J., & Hoffmann, P. (2005). *Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis*. In *HLT/EMNLP 2005*.

---

## 📝 Citation

```
@misc{selfdriving-tweet-sentiment,
  title  = {Sentiment Analysis of Self-Driving Car Tweets},
  author = {Your Name},
  year   = {2025},
  note   = {Course Project}
}
```

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

```
```
