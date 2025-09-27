import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Load data
df_train = pd.read_csv('drive/My Drive/Projects/Twitter Sentiment/train_tweet.csv')
df_test = pd.read_csv('drive/My Drive/Projects/Twitter Sentiment/test_tweets.csv')

print(df_train.shape)
print(df_test.shape)

df_train.head()
df_test.head()

# Null check
df_train.isnull().any()
df_test.isnull().any()

# Peek at negative examples
df_train[df_train['label'] == 0].head(10)

# Peek at positive examples
df_train[df_train['label'] == 1].head(10)

# Class balance
df_train['label'].value_counts().plot.bar(color='pink', figsize=(6, 4))

# Tweet length distributions
dist_len_train = df_train['tweet'].str.len().plot.hist(color='pink', figsize=(6, 4))
dist_len_test = df_test['tweet'].str.len().plot.hist(color='orange', figsize=(6, 4))

# Add length feature
df_train['txt_len'] = df_train['tweet'].str.len()
df_test['txt_len'] = df_test['tweet'].str.len()

df_train.head(10)

# Summary by label
df_train.groupby('label').describe()

# Label vs length
df_train.groupby('txt_len').mean()['label'].plot.hist(color='black', figsize=(6, 4))
plt.title('Variation of length')
plt.xlabel('Length')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

# Bag of words (global frequency)
count_vec = CountVectorizer(stop_words='english')
bow_mat = count_vec.fit_transform(df_train.tweet)

bow_sum = bow_mat.sum(axis=0)

token_freq = [(token, bow_sum[0, idx]) for token, idx in count_vec.vocabulary_.items()]
token_freq = sorted(token_freq, key=lambda x: x[1], reverse=True)

freq_df = pd.DataFrame(token_freq, columns=['word', 'freq'])

freq_df.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color='blue')
plt.title("Most Frequently Occurring Words - Top 30")

from wordcloud import WordCloud

# Word cloud (all reviews)
wc_all = WordCloud(background_color='white', width=1000, height=1000)\
    .generate_from_frequencies(dict(token_freq))

plt.figure(figsize=(10, 8))
plt.imshow(wc_all)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22)

# Word cloud (neutral/label=0)
neutral_text = ' '.join([text for text in df_train['tweet'][df_train['label'] == 0]])
wc_neutral = WordCloud(width=800, height=500, random_state=0, max_font_size=110).generate(neutral_text)
plt.figure(figsize=(10, 7))
plt.imshow(wc_neutral, interpolation="bilinear")
plt.axis('off')
plt.title('The Neutral Words')
plt.show()

# Word cloud (negative/label=1)
negative_text = ' '.join([text for text in df_train['tweet'][df_train['label'] == 1]])
wc_negative = WordCloud(background_color='cyan', width=800, height=500, random_state=0, max_font_size=110)\
    .generate(negative_text)
plt.figure(figsize=(10, 7))
plt.imshow(wc_negative, interpolation="bilinear")
plt.axis('off')
plt.title('The Negative Words')
plt.show()

# Hashtag extraction
def hashtag_extract(arr):
    tags = []
    for s in arr:
        ht = re.findall(r"#(\w+)", s)
        tags.append(ht)
    return tags

# From neutral tweets
HT_regular = hashtag_extract(df_train['tweet'][df_train['label'] == 0])

# From negative tweets
HT_negative = hashtag_extract(df_train['tweet'][df_train['label'] == 1])

# Flatten lists
HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

a_reg = nltk.FreqDist(HT_regular)
top_reg_df = pd.DataFrame({'Hashtag': list(a_reg.keys()), 'Count': list(a_reg.values())})

# Top 20 neutral hashtags
top_reg_df = top_reg_df.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=top_reg_df, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

a_neg = nltk.FreqDist(HT_negative)
top_neg_df = pd.DataFrame({'Hashtag': list(a_neg.keys()), 'Count': list(a_neg.values())})

# Top 20 negative hashtags
top_neg_df = top_neg_df.nlargest(columns="Count", n=20)
plt.figure(figsize=(16, 5))
ax = sns.barplot(data=top_neg_df, x="Hashtag", y="Count")
ax.set(ylabel='Count')
plt.show()

# Tokenize tweets
tokenized_toks = df_train['tweet'].apply(lambda x: x.split())

# Word2Vec setup
import gensim
w2v = gensim.models.Word2Vec(
    tokenized_toks,
    size=200,         # embedding size
    window=5,         # context window
    min_count=2,
    sg=1,             # skip-gram
    hs=0,
    negative=10,      # negative sampling
    workers=2,
    seed=34
)

w2v.train(tokenized_toks, total_examples=len(df_train['tweet']), epochs=20)

# Explore neighbors
w2v.wv.most_similar(positive="dinner")
w2v.wv.most_similar(positive="cancer")
w2v.wv.most_similar(positive="apple")
w2v.wv.most_similar(negative="hate")

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence

# Label tweets for doc2vec-like usage
def tag_docs(series):
    out = []
    for i, s in zip(series.index, series):
        out.append(LabeledSentence(s, [f"tweet_{i}"]))
    return out

labeled_docs = tag_docs(tokenized_toks)
labeled_docs[:6]

# Text cleaning and stemming
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus_train = []
for i in range(0, 31962):
    review = re.sub('[^a-zA-Z]', ' ', df_train['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # Remove stopwords and apply stemming
    review = [ps.stem(w) for w in review if w not in set(stopwords.words('english'))]
    corpus_train.append(' '.join(review))

corpus_test = []
for i in range(0, 17197):
    review = re.sub('[^a-zA-Z]', ' ', df_test['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    # Remove stopwords and apply stemming
    review = [ps.stem(w) for w in review if w not in set(stopwords.words('english'))]
    corpus_test.append(' '.join(review))

# Bag of Words: train
from sklearn.feature_extraction.text import CountVectorizer

bow_train_vec = CountVectorizer(max_features=2500)
X = bow_train_vec.fit_transform(corpus_train).toarray()
y = df_train.iloc[:, 1]

print(X.shape)
print(y.shape)

# Bag of Words: test
bow_test_vec = CountVectorizer(max_features=2500)
X_test = bow_test_vec.fit_transform(corpus_test).toarray()

print(X_test.shape)

# Train/validation split
from sklearn.model_selection import train_test_split

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_tr.shape)
print(X_va.shape)
print(y_tr.shape)
print(y_va.shape)

# Standardize features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_va = scaler.transform(X_va)
X_test = scaler.transform(X_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

rf_clf = RandomForestClassifier()
rf_clf.fit(X_tr, y_tr)
y_pred = rf_clf.predict(X_va)
print("Training Accuracy :", rf_clf.score(X_tr, y_tr))
print("Validation Accuracy :", rf_clf.score(X_va, y_va))
print("F1 score :", f1_score(y_va, y_pred))
cm = confusion_matrix(y_va, y_pred)
print(cm)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X_tr, y_tr)
y_pred = log_clf.predict(X_va)
print("Training Accuracy :", log_clf.score(X_tr, y_tr))
print("Validation Accuracy :", log_clf.score(X_va, y_va))
print("f1 score :", f1_score(y_va, y_pred))
cm = confusion_matrix(y_va, y_pred)
print(cm)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_tr, y_tr)
y_pred = dt_clf.predict(X_va)
print("Training Accuracy :", dt_clf.score(X_tr, y_tr))
print("Validation Accuracy :", dt_clf.score(X_va, y_va))
print("f1 score :", f1_score(y_va, y_pred))
cm = confusion_matrix(y_va, y_pred)
print(cm)

# SVM
from sklearn.svm import SVC

svc_clf = SVC()
svc_clf.fit(X_tr, y_tr)
y_pred = svc_clf.predict(X_va)
print("Training Accuracy :", svc_clf.score(X_tr, y_tr))
print("Validation Accuracy :", svc_clf.score(X_va, y_va))
print("f1 score :", f1_score(y_va, y_pred))
cm = confusion_matrix(y_va, y_pred)
print(cm)

# XGBoost
from xgboost import XGBClassifier

xgb_clf = XGBClassifier()
xgb_clf.fit(X_tr, y_tr)
y_pred = xgb_clf.predict(X_va)
print("Training Accuracy :", xgb_clf.score(X_tr, y_tr))
print("Validation Accuracy :", xgb_clf.score(X_va, y_va))
print("f1 score :", f1_score(y_va, y_pred))
cm = confusion_matrix(y_va, y_pred)
print(cm)
