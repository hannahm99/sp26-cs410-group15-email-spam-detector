import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Placeholder random data to text code
df = pd.read_csv("./data/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])


def preprocess(text):
    text = text.lower()                          
    text = re.sub(r"[^a-z\s]", "", text)         
    text = re.sub(r"\s+", " ", text).strip()    
    return text

df["clean_text"] = df["text"].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1, 2),  
    stop_words="english",
    sublinear_tf=True       
)
 
X = vectorizer.fit_transform(df["clean_text"])
y = (df["label"] == "spam").astype(int) 
 
 
# Train-test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
 
# Define models with basic hyperparamters (will be tuned later)
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=42
    ),
    "SVM": LinearSVC(
        C=1.0, max_iter=2000, random_state=42
    ),
}
 
# Calculate basic metrics and generate confusion matrices for each model
results = {} 
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
 
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
 
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
 
    results[name] = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std":  cv_scores.std(),
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }
 
    print(f"\n--- {name} ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of emails flagged as spam, how many truly are)")
    print(f"  Recall    : {rec:.4f}  (of all spam, how much we catch)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")
 
 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices — Spam Detection Models", fontsize=14, fontweight="bold")
 
for ax, (name, res) in zip(axes, results.items()):
    cm = res["confusion_matrix"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax, cbar=False
    )
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
 
plt.tight_layout()
plt.savefig("./results/confusion_matrices.png", dpi=150, bbox_inches="tight")
 
metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
model_names = list(results.keys())
 
x = np.arange(len(metrics))
width = 0.25
colors = ["#FF0000", "#A903BC", "#1100FF"]
 
fig, ax = plt.subplots(figsize=(10, 5))
 
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5
        )
 
ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Accuracy, Precision, Recall, F1", fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.08)
ax.legend()
ax.grid(axis="y", alpha=0.3)
 
plt.tight_layout()
plt.savefig("./results/model_comparison.png", dpi=150, bbox_inches="tight")
 
 
summary = pd.DataFrame({
    name: {
        "Accuracy":  f"{res['accuracy']:.4f}",
        "Precision": f"{res['precision']:.4f}",
        "Recall":    f"{res['recall']:.4f}",
        "F1-Score":  f"{res['f1']:.4f}",
        "CV F1 (mean±std)": f"{res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f}",
    }
    for name, res in results.items()
}).T