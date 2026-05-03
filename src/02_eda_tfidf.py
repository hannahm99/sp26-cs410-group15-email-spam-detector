import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("outputs/figures", exist_ok=True)

# read preprocessed data from previous output
df = pd.read_csv("results/preprocessed.csv")

# remove NaN rows
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str) 

# print labels and message length
print("Table 1: Label Counts")
print(df["label"].value_counts())

df["msg_len"] = df["text"].apply(lambda x: len(x.split()))

print("Table 2: Message Length Stats")
print(df.groupby("label")["msg_len"].describe())

# plot message length and spam-ham distribution
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.boxplot(x="label", y="msg_len", data=df)
plt.title("Message Length Boxplot")

plt.subplot(1,2,2)
sns.histplot(data=df, x="msg_len", hue="label", bins=50, kde=True)
plt.title("Message Length Histogram")

plt.tight_layout()
plt.savefig("outputs/figures/fig2_msg_length.png")
plt.close()

df["label"].value_counts().plot(kind="bar")
plt.title("Spam vs Ham Distribution")
plt.xticks([0,1], ["Ham (0)", "Spam (1)"], rotation=0)
plt.ylabel("Count")

plt.savefig("outputs/figures/fig1_label_dist.png")
plt.close()

# keep track of top words in spam vs ham and plot them
def get_top_words(text_series, n=20):
    all_words = " ".join(text_series).split()
    return Counter(all_words).most_common(n)

spam_words = get_top_words(df[df["label"] == 1]["text"])
ham_words = get_top_words(df[df["label"] == 0]["text"])

spam_df = pd.DataFrame(spam_words, columns=["word", "count"])
ham_df = pd.DataFrame(ham_words, columns=["word", "count"])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(data=spam_df, x="count", y="word")
plt.title("Top Spam Words")

plt.subplot(1,2,2)
sns.barplot(data=ham_df, x="count", y="word")
plt.title("Top Ham Words")

plt.tight_layout()
plt.savefig("outputs/figures/fig3_word_freq.png")
plt.close()

# compare vocabulary size
spam_vocab = set(" ".join(df[df["label"]==1]["text"]).split())
ham_vocab = set(" ".join(df[df["label"]==0]["text"]).split())

plt.bar(["Ham", "Spam"], [len(ham_vocab), len(spam_vocab)])
plt.title("Vocabulary Size Comparison")

plt.savefig("outputs/figures/fig4_vocab_size.png")
plt.close()

# analyze bigrams in spam messages and plot top 20
def get_bigrams(text_series, n=20):
    bigrams = []
    for text in text_series:
        words = text.split()
        bigrams += [" ".join([words[i], words[i+1]]) for i in range(len(words)-1)]
    return Counter(bigrams).most_common(n)

spam_bigrams = get_bigrams(df[df["label"] == 1]["text"])
bigram_df = pd.DataFrame(spam_bigrams, columns=["bigram", "count"])

plt.figure(figsize=(8,6))
sns.barplot(data=bigram_df, x="count", y="bigram")
plt.title("Top Spam Bigrams")

plt.tight_layout()
plt.savefig("outputs/figures/fig5_bigrams.png")
plt.close()

# split data for TF-IDF vectorization and model training
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# perform TF-IDF vectorization with unigrams and bigrams, sublinear TF scaling, and max 5000 features
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    sublinear_tf=True,
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# save vectorizer and train/test splits for use in model training and analysis
with open("outputs/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("outputs/train_test_splits.pkl", "wb") as f:
    pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), f)

# analyze top TF-IDF features for spam vs ham and plot them
feature_names = np.array(vectorizer.get_feature_names_out())

spam_indices = (y_train == 1).values
ham_indices = (y_train == 0).values

spam_tfidf_mean = X_train_tfidf[spam_indices].mean(axis=0).A1
ham_tfidf_mean = X_train_tfidf[ham_indices].mean(axis=0).A1

top_spam = feature_names[np.argsort(spam_tfidf_mean)[-20:]]
top_ham = feature_names[np.argsort(ham_tfidf_mean)[-20:]]

# print top TF-IDF features for spam vs ham
print("Top Spam TF-IDF Features")
print(top_spam)

print("Top Ham TF-IDF Features")
print(top_ham)