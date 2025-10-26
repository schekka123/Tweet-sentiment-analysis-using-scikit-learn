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

---

## 🔬 Model Details & Feature Engineering

### Preprocessing Pipeline

A common pipeline was applied before feature extraction for most models:
`Raw Tweet` → `Sentence Split` → `Word Tokenize` → `POS Tag` → `Filter (Keep Noun/Adj/Verb)` → `Lemmatize` → `Processed Text String`

### 1. Bag-of-Words (BoW)

- **Input:** Processed Text String
- **Vectorization:** `TfidfVectorizer` (from `scikit-learn`)
- **N-grams Tested:** `(1, 1)`, `(1, 2)`, `(1, 3)`
- **Classifier:** `LinearSVC`

### 2. MPQA Lexicon Features

- **Input:** Tokenized Tweet (before extensive filtering)
- **Feature Extraction:** Counts occurrences of positive, negative, and neutral words using the [MPQA Subjectivity Lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/).
- **Vectorization:** `DictVectorizer` (converts `{'pos': p, 'neg': n, 'neu': z}` counts to matrix).
- **Classifier:** `LinearSVC`

### 3. Hybrid (BoW + MPQA)

- **Input:** Processed Text String + MPQA Counts
- **Feature Engineering:** Appends string tokens (e.g., `_POS_ _POS_` if 2 positive words found) to the end of the processed text.
- **Vectorization:** `TfidfVectorizer` on the augmented string.
- **Classifier:** `LinearSVC`

### 4. Dependency Triples

- **Input:** Raw Tweet
- **Parsing:** Uses `spaCy` (or `NLTK` + MaltParser) to generate dependency parses for each sentence.
- **Feature Extraction:** Extracts `(head.lemma_, dep_relation, child.lemma_)` triples.
- **Vectorization:** Converts triples to strings (e.g., `"car_nsubj_be"`) and uses `TfidfVectorizer` or `HashingVectorizer`.
- **Classifier:** `LinearSVC`

---

## 🚀 Usage

1.  **Run the complete pipeline** (preprocess, train all models, evaluate, and generate summary):
    ```bash
    python main.py --data_path data/sentiment-self-driving-cars.csv --output_dir results/
    ```

2.  **Or, run steps individually:**
    ```bash
    # Preprocess data
    python src/preprocess.py --input data/sentiment-self-driving-cars.csv --output data/processed_tweets.pkl

    # Train and evaluate a single model
    python src/train.py --data data/processed_tweets.pkl --model hybrid --save_report results/hybrid_report.json

    # Generate summary plots and tables from results
    python src/visualize.py --results_dir results/ --output_dir results/
    ```

---

## 📈 Results & Discussion

### Performance Summary

The performance of each custom model was benchmarked against Naive Bayes and the VADER sentiment library. The combined **Bag-of-Words + MPQA Lexicon** approach achieved the highest accuracy.

| Model                                | Accuracy Score |
| :----------------------------------- | :------------- |
| **🥇 BoW + Subjectivity Lexicon** | **0.648** |
| BoW (Unigrams only)                  | 0.637 |
| BoW (Unigrams + Bigrams)             | 0.620 |
| BoW (Unigrams + Bigrams + Trigrams)  | 0.610 |
| Dependency Triples                   | 0.584 |
| VADER (Baseline)                     | 0.569 |
| Naive Bayes (Baseline)               | 0.550 |
| Subjectivity Lexicon Only            | 0.549 |

### Key Insights

* **Hybrid is Best**: Combining traditional BoW features with explicit sentiment cues from a lexicon provided the most effective model.
* **BoW is a Strong Performer**: A simple, well-processed Bag-of-Words model significantly outperformed both baselines, showing its strength even on noisy tweet data.
* **More N-grams != Better**: Contrary to expectations, adding bigrams and trigrams slightly degraded performance, possibly due to overfitting on this dataset.
* **Lexicon Alone is Insufficient**: The lexicon-only model performed the worst, likely because it couldn't handle the slang, misspellings, and novel words common in tweets.
* **Syntax is secondary**: Dependency triples, which capture grammatical structure, were better than the baselines but less effective than the lexical BoW approaches for this task.

---

## 💡 Future Work

While the project yielded a moderately successful model, several avenues exist for improvement:

1.  **Word Embeddings**: Replace TF-IDF with pre-trained word embeddings (e.g., GloVe, Word2Vec, fastText) to capture semantic meaning.
2.  **Transformer Models**: Fine-tune a pre-trained transformer model like `BERTweet` or `DistilBERT`, which are state-of-the-art for many NLP tasks.
3.  **Feature Engineering**: Explicitly model features for emojis, hashtags, user mentions, negation, and sentiment shifters (e.g., "very", "hardly").
4.  **Error Analysis**: Conduct a deep dive into misclassified examples to identify systematic model weaknesses.
5.  **Ensemble Methods**: Combine the predictions of the best-performing models to potentially boost overall accuracy.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📚 References

- Brownlee, J. (2019). *A gentle introduction to the bag-of-words model*. Machine Learning Mastery.
- CrowdFlower. (2016). *Sentiment self-driving cars - dataset by Crowdflower*. data.world.
- Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM-14.
- Manning, C., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Nivre, J., Hall, J., & Nilsson, J. (2006). *Maltparser: A data-driven parser-generator for dependency parsing*. LREC.
- Wilson, T., Wiebe, J., & Hoffmann, P. (2005). *Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis*. HLT-EMNLP-2005.
