# Twitter-Sentiment-Analysis

A practical Natural Language Processing project for classifying tweet sentiment. The pipeline covers text cleaning, exploratory analysis, feature engineering, and supervised learning models to distinguish non‑toxic tweets from toxic ones.

## Objective

Detect hate speech in tweets. Label `1` denotes a racist/sexist tweet, and label `0` denotes not racist/sexist. Models are evaluated primarily with F1‑score.

## Dataset

* `train_tweet.csv`: training data with text and label columns
* `test_tweets.csv`: test data with text but no labels

> Paths in the example code assume: `drive/My Drive/Projects/Twitter Sentiment/`.

## Methods

1. **Preprocessing**

   * Lowercasing, non‑alphabetic removal, tokenization
   * Stopword removal and Porter stemming (NLTK)
   * Optional hashtag extraction using regex
2. **Exploration**

   * Class balance visualization
   * Tweet length distributions
   * Most frequent tokens via `CountVectorizer`
   * Word clouds for overall, neutral (label=0), and negative (label=1) subsets
   * Top hashtags for each subset
3. **Feature Engineering**

   * Bag‑of‑Words (max_features=2500)
   * Optional Word2Vec (Gensim) for semantic inspection
4. **Modeling**

   * Train/validation split (25%, random_state=42)
   * Standardization on BoW inputs
   * Models: RandomForest, LogisticRegression, DecisionTree, SVC, XGBoost
   * Metrics: Accuracy and F1‑score with confusion matrix

## Project Structure

```
Twitter-Sentiment-Analysis/
├─ data/
│  ├─ train_tweet.csv
│  └─ test_tweets.csv
├─ notebooks/
│  └─ exploration.ipynb
├─ src/
│  └─ sentiment_pipeline.py
└─ README.md
```

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim wordcloud xgboost tqdm
```

### 2) NLTK resources

```python
import nltk
nltk.download('stopwords')
```

### 3) Run the analysis

You can place the provided script (from this repo’s code section) into `src/sentiment_pipeline.py` and execute:

```bash
python src/sentiment_pipeline.py
```

This will load the data, preprocess text, create features, train baseline models, and report validation metrics.

## Key Code Highlights

* **Loading and inspection**

  * Shape checks, `head()` previews, null checks
* **EDA**

  * Class distribution bar plot
  * Length histograms for train/test
  * Token frequency bar chart (top 30)
  * Hashtag frequency by label (top 20)
* **Text preparation**

  * Regex cleaning, stopword removal, stemming
* **Features**

  * BoW with `CountVectorizer(max_features=2500)`
  * Optional Word2Vec training for similarity queries
* **Models and metrics**

  * RandomForest, LogisticRegression, DecisionTree, SVC, XGBClassifier
  * Validation accuracy and F1‑score
  * Confusion matrix

## Usage Notes

* Ensure the dataset path is correct for your environment (local vs cloud drive).
* Word cloud generation is optional if you run in a non‑GUI environment.
* XGBoost requires the `xgboost` package; skip or install as needed.

## Results

Typical outputs include validation accuracy and F1‑score per model. Use these to compare model trade‑offs and pick a final candidate for deployment.

## Extending

* Add TF‑IDF features and compare with BoW
* Try class‑weighting or resampling to address class imbalance
* Hyperparameter search with `GridSearchCV` or `Optuna`
* Replace bag‑of‑words with embeddings (e.g., fastText, GloVe) or transformer encoders

## References

* scikit‑learn documentation
* NLTK corpora and preprocessing utilities
* Gensim Word2Vec
* XGBoost Python API

## License

MIT License. See `LICENSE` if provided.
