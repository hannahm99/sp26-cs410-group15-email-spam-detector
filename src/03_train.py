import os
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

os.makedirs("outputs", exist_ok=True)

# Load TF-IDF train/test split
with open("outputs/train_test_splits.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Training label counts:")
print(y_train.value_counts().sort_index())
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = LinearSVC(max_iter=2000, random_state=42)

print("Cross-Validation Results")

nb_scores = cross_val_score(nb, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
svm_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)

print(f"Naive Bayes: CV F1 = {nb_scores.mean():.4f} +- {nb_scores.std():.4f}")
print(f"Logistic Regression: CV F1 = {lr_scores.mean():.4f} +- {lr_scores.std():.4f}")
print(f"Linear SVM: CV F1 = {svm_scores.mean():.4f} +- {svm_scores.std():.4f}")
print()

# Grid search for LR and SVM
c_values = [0.01, 0.1, 1, 10]

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    {"C": c_values},
    cv=cv,
    scoring="f1",
    n_jobs= -1
)
lr_grid.fit(X_train, y_train)

svm_grid = GridSearchCV(
    LinearSVC(max_iter=2000, random_state=42),
    {"C": c_values},
    cv=cv,
    scoring ="f1",
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)

print("Grid Search Results")
print(f"Logistic Regression best C: {lr_grid.best_params_['C']}")
print(f"Logistic Regression best CV F1: {lr_grid.best_score_:.4f}")
print(f"Linear SVM best C: {svm_grid.best_params_['C']}")
print(f"Linear SVM best CV F1: {svm_grid.best_score_:.4f}")
print()

# Final models trained on full training set
nb.fit(X_train, y_train)
best_lr = lr_grid.best_estimator_
best_svm = svm_grid.best_estimator_

model_scores = {
    "Naive Bayes": nb_scores.mean(),
    "Logistic Regression": lr_grid.best_score_,
    "Linear SVM": svm_grid.best_score_
}

models = {
    "Naive Bayes": nb,
    "Logistic Regression": best_lr,
    "Linear SVM": best_svm
}

best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

# Save all models
with open("outputs/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb, f)

with open("outputs/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(best_lr, f)

with open("outputs/linear_svm_model.pkl", "wb") as f:
    pickle.dump(best_svm, f)

with open("outputs/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

training_summary = {
    "Naive Bayes": {
        "cv_f1_mean": nb_scores.mean(),
        "cv_f1_std": nb_scores.std(),
        "best_params": {}
    },
    "Logistic Regression": {
        "cv_f1_mean": lr_grid.best_score_,
        "cv_f1_std": lr_scores.std(),
        "best_params": lr_grid.best_params_
    },
    "Linear SVM": {
        "cv_f1_mean": svm_grid.best_score_,
        "cv_f1_std": svm_scores.std(),
        "best_params": svm_grid.best_params_
    },
    "best_model_name": best_model_name
}

with open("outputs/training_summary.pkl", "wb") as f:
    pickle.dump(training_summary, f)

print(f"Best model: {best_model_name} with CV F1 = {model_scores[best_model_name]:.4f}")