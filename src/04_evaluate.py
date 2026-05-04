import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


MODEL_FILES = {
    "Naive Bayes": "outputs/naive_bayes_model.pkl",
    "Logistic Regression": "outputs/logistic_regression_model.pkl",
    "Linear SVM": "outputs/linear_svm_model.pkl",
}


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_splits(splits_path):
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"Could not find '{splits_path}'. Please run 02_eda_tfidf.py first to generate the train/test split."
        )
    with open(splits_path, "rb") as f:
        return pickle.load(f)


def load_models():
    models = {}
    missing = [name for name, path in MODEL_FILES.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing trained model files: {}. Please run 03_train.py first.".format(
                ", ".join(missing)
            )
        )
    for name, path in MODEL_FILES.items():
        models[name] = load_pickle(path)
    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"],
        digits=4,
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_spam": precision_score(y_test, y_pred, pos_label=1),
        "recall_spam": recall_score(y_test, y_pred, pos_label=1),
        "f1_spam": f1_score(y_test, y_pred, pos_label=1),
    }
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    metrics.update(
        {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "report": report,
            "y_pred": y_pred,
        }
    )
    return metrics


def build_metrics_table(results):
    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision (spam)": metrics["precision_spam"],
                "Recall (spam)": metrics["recall_spam"],
                "F1 (spam)": metrics["f1_spam"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
            }
        )
    df = pd.DataFrame(rows)
    df = df[
        [
            "Model",
            "Accuracy",
            "Precision (spam)",
            "Recall (spam)",
            "F1 (spam)",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
        ]
    ]
    return df


def plot_confusion_matrices(results, y_test, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    model_names = list(results.keys())

    for ax, model_name in zip(axes, model_names):
        cm = confusion_matrix(y_test, results[model_name]["y_pred"], labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticklabels(["ham", "spam"])
        ax.set_yticklabels(["ham", "spam"], rotation=0)

    fig.suptitle("Confusion Matrices for Spam Classification", fontsize=16)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = ["Accuracy", "Precision (spam)", "Recall (spam)", "F1 (spam)"]
    plot_df = df.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1.0)
    plt.title("Model Comparison on Held-Out Test Set")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.legend(loc="lower right")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_test_texts_if_available():
    candidate_paths = ["outputs/preprocessed.csv", "results/preprocessed.csv"]
    for path in candidate_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "text" not in df.columns or "label" not in df.columns:
                continue
            df = df.dropna(subset=["text"]).copy()
            df["text"] = df["text"].astype(str)
            return df
    return None


def print_misclassified_examples(
    best_model_name, best_model, X_test_matrix, y_test, y_test_texts, output_path=None
):
    if y_test_texts is None:
        print(
            "Warning: Could not load preprocessed text data from outputs/preprocessed.csv or results/preprocessed.csv."
        )
        print(
            "The script can still evaluate models, but cannot print the raw misclassified text without the preprocessed CSV."
        )
        return

    y_pred = best_model.predict(X_test_matrix)
    false_positive_mask = (y_test == 0) & (y_pred == 1)
    false_positives = y_test_texts[false_positive_mask]

    print("Misclassified Ham Messages Flagged as Spam")
    print("------------------------------------------")
    print(f"Using model: {best_model_name}\n")

    num_false_positives = len(false_positives)
    if num_false_positives > 0:
        count = min(3, num_false_positives)
        sample_texts = y_test_texts[false_positive_mask].head(count)
        for idx, message in enumerate(sample_texts, start=1):
            print(f"Example {idx}")
            print("True label: ham")
            print("Predicted label: spam")
            print(f"Message: {message}\n")
        return

    print("No ham messages were flagged as spam by the selected model.")
    false_negative_mask = (y_test == 1) & (y_pred == 0)
    if false_negative_mask.sum() > 0:
        print("Showing up to 3 spam messages incorrectly predicted as ham instead:\n")
        sample_texts = y_test_texts[false_negative_mask].head(3)
        for idx, message in enumerate(sample_texts, start=1):
            print(f"Example {idx}")
            print("True label: spam")
            print("Predicted label: ham")
            print(f"Message: {message}\n")
    else:
        print("No misclassified examples are available for the selected model.")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    splits_path = os.path.join(output_dir, "train_test_splits.pkl")
    X_train, X_test, y_train, y_test = load_splits(splits_path)
    print("Loaded held-out test data from: {}".format(splits_path))
    print("X_test shape: {}".format(X_test.shape))
    print("y_test distribution:\n{}".format(y_test.value_counts().sort_index()))
    print()

    models = load_models()
    print("Loaded trained models from outputs directory.")
    print()

    results = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics
        print(f"Classification Report: {model_name}")
        print("-" * (23 + len(model_name)))
        print(metrics["report"])

    metrics_df = build_metrics_table(results)
    csv_path = os.path.join(output_dir, "model_metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False)

    print("Summary Metrics Table")
    print(metrics_df.to_string(index=False, float_format="{:.4f}".format))
    print(f"Saved metrics summary to: {csv_path}\n")

    cm_path = os.path.join(figures_dir, "fig6_confusion_matrices.png")
    plot_confusion_matrices(results, y_test, cm_path)
    print(f"Saved confusion matrices figure to: {cm_path}")

    comparison_path = os.path.join(figures_dir, "fig7_model_comparison.png")
    plot_model_comparison(metrics_df, comparison_path)
    print(f"Saved model comparison figure to: {comparison_path}\n")

    best_model_name = None
    best_model = None

    summary_path = os.path.join(output_dir, "training_summary.pkl")
    if os.path.exists(summary_path):
        training_summary = load_pickle(summary_path)
        best_model_name = training_summary.get("best_model_name")
        if best_model_name in models:
            best_model = models[best_model_name]
            print(f"Best model based on training summary: {best_model_name}")
    if best_model is None:
        # choose best model using held-out spam F1, tie-break by spam precision and fixed priority
        sorted_models = sorted(
            results.items(),
            key=lambda item: (
                item[1]["f1_spam"],
                item[1]["precision_spam"],
                0 if item[0] == "Logistic Regression" else 1 if item[0] == "Linear SVM" else 2,
            ),
            reverse=True,
        )
        best_model_name, _ = sorted_models[0]
        best_model = models[best_model_name]
        print(f"Best model based on held-out spam F1: {best_model_name}")

    preprocessed_df = load_test_texts_if_available()
    if preprocessed_df is not None:
        X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
            preprocessed_df["text"],
            preprocessed_df["label"],
            test_size=0.2,
            stratify=preprocessed_df["label"],
            random_state=42,
        )
        text_series = X_text_test.reset_index(drop=True)
        y_test_texts = text_series
    else:
        y_test_texts = None

    print_misclassified_examples(best_model_name, best_model, X_test, y_test.reset_index(drop=True), y_test_texts)


if __name__ == "__main__":
    main()
