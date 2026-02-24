# 🐦 Twitter Hate Speech Detection
### An NLP classification pipeline that distinguishes racist and sexist tweets from general content - built to demonstrate real-world text preprocessing, class-imbalance awareness, and multi-model benchmarking.

> *A business-driven NLP project focused on measurable harm-reduction impact through scalable text classification and interpretable feature engineering.*

---

## 💼 Business Problem

Social platforms face a mounting regulatory and reputational crisis around harmful content. In 2023, the EU's Digital Services Act came into force requiring large platforms to actively detect and remove illegal hate speech. In the US, advertiser boycotts of platforms perceived as unsafe have cost hundreds of millions in lost revenue.

Moderation at scale is not a human problem. Twitter processes hundreds of millions of tweets per day. No team of human reviewers can keep pace. Automated detection is not optional - it is infrastructure.

**The question this project answers:**
> *Can we build a supervised classifier that reliably identifies racist and sexist tweets using only the text of the tweet itself, with no metadata, no user history, and no engagement signals?*

---

## ❗ Why This Matters

- **Regulatory exposure.** Platforms that fail to meet hate speech removal thresholds under the EU DSA face fines of up to 6% of global annual revenue. For a major platform, that is billions of dollars.
- **Advertiser trust.** Brand safety is a top-three concern for digital advertisers. A single viral controversy about ads appearing next to hate speech can trigger platform-wide advertiser pullouts.
- **User retention.** Marginalized users who encounter unchecked hate speech disengage. Every user who leaves due to a hostile environment represents compounding lost revenue.
- **Reviewer wellbeing.** Human moderators reviewing hate speech at scale face documented psychological harm. An accurate first-pass automated classifier reduces the volume of content that ever reaches a human reviewer.

A classifier that correctly flags even 70-80% of racist and sexist tweets before they spread reduces harm, reduces liability, and reduces operational cost simultaneously.

---

## 🎯 Objective

Build an end-to-end binary text classification pipeline that:

1. Cleans and preprocesses raw tweet text (regex, stemming, stopword removal)
2. Explores the data to understand class distribution, vocabulary patterns, and hashtag usage by label
3. Trains Word2Vec embeddings to validate that the learned semantic space reflects tweet context
4. Engineers Bag-of-Words features and benchmarks five classifier families
5. Evaluates all models on F1-score as the primary metric (not accuracy) to account for class imbalance

---

## 💰 Business Impact

*Estimates are modeled against realistic content moderation industry benchmarks to illustrate production-scale value.*

| Impact Area | Estimate | Basis |
|---|---|---|
| **Moderation throughput** | 17,000+ tweets processed per batch run | Test set size; production systems scale to millions per day |
| **Human review reduction** | 70-80% reduction in content reaching reviewers | Accurate first-pass filter routes only borderline cases to humans |
| **Regulatory risk mitigation** | Up to 6% global revenue protected | EU DSA fine threshold for non-compliant large platforms |
| **Advertiser safety** | Brand-unsafe impressions significantly reduced | Direct input to ad placement exclusion lists |
| **Reviewer wellbeing** | Fewer staff exposed to high-volume hate speech | Documented benefit in Trust and Safety literature |
| **Response latency** | Near real-time classification at inference | Lightweight BoW plus linear model runs in milliseconds per tweet |

> **Note:** This is a portfolio project using a public dataset of 31,962 labeled tweets. Business impact figures are modeled assumptions illustrating production-scale value, not claims from a live deployment.

---

## 📊 Dataset Overview

| File | Rows | Columns |
|---|---|---|
| `train_tweet.csv` | 31,962 | `id`, `label`, `tweet` |
| `test_tweets.csv` | 17,197 | `id`, `tweet` |

**Labels:**

| Label | Meaning | Share |
|---|---|---|
| `0` | Non-racist / non-sexist tweet | ~93% |
| `1` | Racist or sexist tweet | ~7% |

**Class imbalance is a core challenge.** A naive model that predicts "not hate speech" for every tweet achieves 93% accuracy while being completely useless. This is exactly why F1-score, not accuracy, is the right evaluation metric here.

---

## 🧠 Methodology and Decision Process

### Why F1-score over accuracy?

This is the most important modeling decision in the project. With ~93% of tweets labeled non-toxic, a model that always predicts 0 achieves 93% accuracy with zero utility. F1-score balances precision and recall, penalizing both false positives (wrongly flagging clean content) and false negatives (missing actual hate speech). For a content moderation system, both error types carry real cost: false positives create censorship complaints and user churn; false negatives create harm and regulatory exposure.

### Why Bag-of-Words over TF-IDF?

For short texts like tweets (typically under 140 characters), TF-IDF's document frequency weighting provides less benefit than on longer documents. Toxic terminology tends to be rare at the corpus level and would be upweighted by TF-IDF anyway. BoW at max_features=2500 provides a clean, interpretable feature space where the presence of specific toxic vocabulary tokens directly drives classification decisions. Interpretability matters here: a Trust and Safety team needs to understand why a tweet was flagged.

### Why build Word2Vec if BoW is the modeling feature?

Word2Vec was used for exploratory validation, not classification. Training embeddings on the tweet corpus and querying `most_similar("hate")` and `most_similar("cancer")` confirms that the model learned contextually meaningful relationships from this specific dataset. This is a diagnostic step: if the semantic space looked random, it would signal that the text was too noisy or the corpus too small to support meaningful NLP. The embeddings passed that check, providing confidence that the cleaned text carries real signal.

### Why StandardScaler on BoW features?

BoW vectors are raw counts - words that appear multiple times in a tweet have higher values than words that appear once. StandardScaler normalizes these across features, which is particularly important for SVC and LogisticRegression. Tree-based models are theoretically scale-invariant, but applying standardization consistently removes scaling as a confounding variable when comparing performance across all five models.

### Why five separate models?

Each model family makes different assumptions and fails in different ways. LogisticRegression provides a fast, interpretable linear baseline that is often competitive on BoW text features. RandomForest handles feature interactions as an ensemble and is less prone to overfitting than a single tree. DecisionTree is an interpretable single-tree baseline that tends to overfit on high-dimensional text without pruning. SVC performs well on sparse high-dimensional data, which is the natural regime of BoW. XGBoost brings gradient boosted trees to compare boosting against RandomForest's bagging. Benchmarking all five in identical conditions reveals which inductive bias fits this problem, not which model is generically "best."

### Why PorterStemmer over lemmatization?

Stemmer is faster and sufficient for the vocabulary patterns in tweet hate speech. The goal is token normalization: "hating", "hated", and "hate" should collapse to the same feature. Lemmatization would require part-of-speech tagging on noisy, informal tweet text - adding complexity without meaningful signal improvement in this context.

---

## 🔍 NLP Pipeline: Step by Step

```
Raw tweet text (31,962 training tweets)
         |
   Regex cleaning - remove non-alphabetic characters
         |
   Lowercase normalization
         |
   Tokenization - str.split()
         |
   Stopword removal - NLTK English stopwords
         |
   PorterStemmer - reduce to root forms
         |
   Rejoin tokens into cleaned string
         |
   CountVectorizer(max_features=2500)
         |
   2500-dimensional BoW vector per tweet
         |
   StandardScaler - normalize feature magnitudes
         |
   Train / validation split (75% / 25%, random_state=42)
         |
   5 classifiers trained and evaluated on F1-score
```

**Separately (exploratory validation only):**
```
Tokenized tweets -> Word2Vec(size=200, window=5, sg=1, epochs=20)
                 -> most_similar() queries for semantic sanity check
```

---

## 📈 Model Results

All models evaluated on the 25% validation split (~7,990 tweets). F1-score is the primary metric.

| Model | Notes |
|---|---|
| Logistic Regression | Strong linear baseline; typically best performer on sparse BoW text |
| Random Forest | Solid ensemble; handles feature interactions; less overfit than single tree |
| Decision Tree | Most prone to overfitting on 2,500-dimensional feature space without pruning |
| SVC | Margin-maximizing objective suits high-dimensional sparse text well |
| XGBoost | Boosting benchmark; compared against RF bagging approach |

> Exact metric values depend on runtime environment and library versions. Run the notebook to reproduce. The relative ordering above reflects expected behavior given model properties and this problem's BoW + class-imbalance structure.

---

## ⚙️ Tech Stack

**Data:** Python, Pandas, NumPy

**NLP:** NLTK (PorterStemmer, stopwords), `re` (regex), scikit-learn (CountVectorizer)

**Embeddings:** Gensim (Word2Vec, Doc2Vec LabeledSentence)

**Modeling:** scikit-learn (LogisticRegression, RandomForest, DecisionTree, SVC, StandardScaler), XGBoost

**Visualization:** Matplotlib, Seaborn, WordCloud, NLTK FreqDist

---

## 🤖 How AI Was Used

| Task | AI-Assisted? | My Role |
|---|---|---|
| Text preprocessing pipeline | No | Built independently |
| Word2Vec configuration | No | Parameters chosen from Gensim documentation |
| Model selection rationale | No | My own benchmarking decision |
| F1 vs. accuracy justification | ChatGPT consulted on imbalance framing | Analysis and articulation mine |
| README business framing | ChatGPT suggested structure | All content, numbers, and analysis written by me |
| Hashtag extraction regex | No | Standard pattern, independently applied |

**Principle:** AI accelerated specific framing decisions. Every preprocessing, modeling, and analytical choice was made and validated independently.

---

## 🌍 How This Framework Applies Elsewhere

The core pipeline - clean raw text, engineer BoW features, benchmark classifiers with an imbalance-aware metric - transfers directly across any domain with short, noisy text and binary classification needs:

- **Customer support triage:** Classify incoming tickets as urgent / non-urgent without reading every message. Replace tweet text with support message; replace hate/not-hate labels with escalation labels.
- **Product review moderation:** Flag fake or policy-violating reviews on e-commerce platforms. Same pipeline, different label schema.
- **Email spam and phishing detection:** BoW on email body text with binary spam label. Identical preprocessing and model structure.
- **Job posting compliance:** Detect discriminatory language in job descriptions before they go live. Preprocessing and classification logic carry over directly.
- **Healthcare feedback:** Flag distressing or safety-relevant patient comments in survey data for human follow-up. The F1-first evaluation framing is equally critical here - missing a safety signal is the costly error.

The class-imbalance handling and F1-first evaluation approach is reusable in any problem where one class is rare but high-stakes.

---

## 📋 Step-by-Step Reproduction Guide

**1. Data Acquisition**

Obtain `train_tweet.csv` and `test_tweets.csv`. Update file paths in the notebook to match your local or cloud environment.

**2. Environment Setup**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim wordcloud xgboost tqdm
python -c "import nltk; nltk.download('stopwords')"
```

**3. EDA**

Run class distribution bar chart, tweet length histograms for train and test, top-30 token frequency chart, word clouds for overall / label-0 / label-1 subsets, and top-20 hashtags per label using regex and `nltk.FreqDist`.

**4. Word2Vec - Exploratory Only**
```python
import gensim
tokenized_tweet = train['tweet'].apply(lambda x: x.split())
model_w2v = gensim.models.Word2Vec(
    tokenized_tweet, size=200, window=5,
    min_count=2, sg=1, negative=10, workers=2, seed=34
)
model_w2v.train(tokenized_tweet, total_examples=len(train['tweet']), epochs=20)
model_w2v.wv.most_similar(positive="hate")
```

**5. Text Preprocessing**
```python
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop = set(stopwords.words('english'))

def clean_corpus(data, n_rows):
    corpus = []
    for i in range(n_rows):
        review = re.sub('[^a-zA-Z]', ' ', data['tweet'][i])
        review = review.lower().split()
        review = [ps.stem(w) for w in review if w not in stop]
        corpus.append(' '.join(review))
    return corpus

train_corpus = clean_corpus(train, 31962)
test_corpus  = clean_corpus(test, 17197)
```

**6. Feature Engineering**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(train_corpus).toarray()
y = train.iloc[:, 1]

# Important: use transform (not fit_transform) on test data
x_test = cv.transform(test_corpus).toarray()
```

**7. Train / Validation Split and Scaling**
```python
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(
    x, y, test_size=0.25, random_state=42
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid  = sc.transform(x_valid)
x_test   = sc.transform(x_test)
```

**8. Model Training and Evaluation**
```python
from sklearn.metrics import f1_score, confusion_matrix

def evaluate(model, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    print(f"\n{name}")
    print(f"  Train Acc : {model.score(x_train, y_train):.4f}")
    print(f"  Valid Acc : {model.score(x_valid, y_valid):.4f}")
    print(f"  F1 Score  : {f1_score(y_valid, y_pred):.4f}")
    print(confusion_matrix(y_valid, y_pred))
```

Run for each model: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, SVC, XGBClassifier.

---

## 📖 Context

**Project type:** Independent portfolio project  
**Role:** Solo end-to-end - EDA, NLP preprocessing, embedding exploration, feature engineering, model benchmarking  
**Stakeholder simulation:** Designed as a first-pass content moderation classifier that a Trust and Safety team could use to reduce human review queue volume  
**Constraints:** Text features only - no user metadata, follower counts, engagement signals, or account history that a production moderation system would have

---

## 💡 Key Learnings

**What I would improve:**

- **Fix the data leakage in test vectorization.** The notebook calls `cv.fit_transform(test_corpus)` when it should call `cv.transform(test_corpus)` after fitting on training data only. In production this causes the test vocabulary to diverge from the training vocabulary, silently degrading model performance on unseen data. Noted in the reproduction guide above with the corrected version.
- **Address class imbalance explicitly.** With a 93% / 7% split, adding `class_weight='balanced'` to sklearn classifiers or applying SMOTE oversampling on the minority class would likely improve F1 meaningfully without any other change.
- **Cross-validation over a single split.** A single 75/25 split produces point estimates. Stratified k-fold CV would give confidence intervals and confirm results are not split-dependent artifacts.
- **TF-IDF as next step.** TF-IDF is the natural comparison point after BoW and would likely improve performance on toxic vocabulary that is rare corpus-wide but strongly predictive when present.
- **Transformer-based classifier.** A fine-tuned BERTweet (BERT pre-trained on tweet data) would substantially outperform BoW approaches by capturing context, negation, and slang that bag-of-words cannot represent.

**What surprised me:**

- Hashtag analysis revealed that toxic and non-toxic tweets occupy largely non-overlapping hashtag spaces. Top hashtags for label-1 tweets clustered tightly around specific political and demographic targets. This suggests hashtag features alone could function as a lightweight, interpretable pre-filter before running the full NLP pipeline.
- Word2Vec semantic neighbors for ambiguous words like "apple" and "dinner" returned contextually sensible results despite the noisy, informal nature of tweet text. This confirmed the preprocessing was clean enough to train meaningful embeddings.
- StandardScaler on BoW features is uncommon (most practitioners skip it for sparse text) but is necessary for stable SVC convergence. Applying it consistently across all models made the comparison cleaner, even if it was technically unnecessary for tree-based methods.

**Business insight gained:**

The hardest problem in content moderation is not the model - it is defining the label. "Racist or sexist" requires consistent human judgment at annotation time, and label noise in training data directly caps model performance. No model sophistication recovers from ambiguous or inconsistent ground-truth labels. In any real deployment, annotation quality and labeling guidelines are at least as important as the choice of classifier.

---

## 🚀 Future Roadmap

- [ ] Fix test set vectorization leakage (transform only, no refit)
- [ ] Add `class_weight='balanced'` across all sklearn classifiers
- [ ] Stratified k-fold cross-validation for robust metric estimates
- [ ] TF-IDF features benchmarked against BoW
- [ ] SMOTE oversampling on minority class
- [ ] Fine-tuned BERTweet or DistilBERT for transformer baseline
- [ ] Hashtag features as standalone interpretable pre-filter
- [ ] Calibrated probability outputs for confidence-based routing to human review queue

---

## 🤝 Let's Connect

If you are working on content moderation, NLP classification, or Trust and Safety infrastructure, I would enjoy the conversation.

Feedback on the preprocessing choices, the class imbalance handling, or the model selection rationale is especially welcome.

- 💼 [LinkedIn](https://www.linkedin.com/in/1adityakadam)
- 📁 [More Projects](https://www.github.com/1adityakadam)
- 📧 [Email](mailto:askadam@iu.edu)

---

*Built with Python, scikit-learn, NLTK, Gensim, and XGBoost*
