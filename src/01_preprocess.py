# CS 410 Final Project — Text Spam Detector | Group 15
# Preprocessing module (01_preprocess.py)

# Loads the UCI SMS Spam Collection dataset, encodes labels (ham=0, spam=1),
# and runs a preprocessing pipeline: lowercase → strip punctuation/numbers →
# tokenize → remove stopwords → lemmatize
# Saves results/preprocessed.csv for use by Elijah for EDA/TF-IDF and Rabia for training

import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# load raw dataset from tab-separated file, assign column names
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])

# encode labels as binary: ham=0, spam=1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# initialize the stopword list and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # convert text to lowercase
    text = text.lower()

    # strip punctuation and numbers, keeping only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # tokenize the text into individual words
    tokens = word_tokenize(text)

    # remove common stopwords (e.g. 'the', 'a', 'is')
    tokens = [t for t in tokens if t not in stop_words]

    # lemmatize each token to its root form (e.g. 'running' → 'run')
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # rejoin cleaned tokens as a single string and return
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess)

# save preprocessed data for Elijah (EDA/TF-IDF) and Rabia (model training)
df.to_csv('results/preprocessed.csv', index=False)

# print class counts to confirm 4825 ham (0) and 747 spam (1)
print("=== Class counts ===")
print(df['label'].value_counts())

# print before/after sample for one ham and one spam message to verify pipeline
print("\n=== Before/After sample ===")
raw = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'text'])
for i in [0, 2]:
    print(f"\n[{'HAM' if raw['label'][i] == 'ham' else 'SPAM'}]")
    print(f"  BEFORE: {raw['text'][i]}")
    print(f"  AFTER:  {df['text'][i]}")