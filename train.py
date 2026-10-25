
import re
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Create output folders
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# 2. Load dataset
df = pd.read_csv("data/reviews.csv")

df = df[["text_", "label"]].dropna().copy()
df["label"] = df["label"].str.strip().str.upper()
df = df[df["label"].isin(["CG", "OR"])]

print("Original records:", len(df))

# 3. Clean review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["review"] = df["text_"].apply(clean_text)
df = df[df["review"].str.len() > 0]
df = df.drop_duplicates(subset=["review"])

# Label meanings:
# CG = computer-generated
# OR = original
df["target"] = df["label"].map({
    "CG": "fake",
    "OR": "genuine"
})

print("Records after cleaning:", len(df))
print(df["target"].value_counts())

# 4. Split dataset: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    df["review"],
    df["target"],
    test_size=0.4,
    random_state=42,
    stratify=df["target"]
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 5. Define classifiers
algorithms = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        random_state=42,
        n_jobs=-1
    )
}

trained_models = {}
comparison = []

# 6. Train and evaluate each model on validation data
for name, classifier in algorithms.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, pred)
    precision = precision_score(
        y_val, pred, pos_label="fake"
    )
    recall = recall_score(
        y_val, pred, pos_label="fake"
    )
    f1 = f1_score(
        y_val, pred, pos_label="fake"
    )

    comparison.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 7. Compare models
results = pd.DataFrame(comparison)
print("\nMODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    "results/model_comparison.csv",
    index=False
)

# 8. Select best model using validation F1
best_name = results.loc[
    results["F1 Score"].idxmax(), "Algorithm"
]
best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)

# 9. Final evaluation on untouched test data
test_predictions = best_model.predict(X_test)

print("\nFINAL TEST RESULTS")
print(classification_report(
    y_test,
    test_predictions,
    labels=["fake", "genuine"]
))

# 10. Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=["fake", "genuine"],
    cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(
    "results/confusion_matrix.png",
    dpi=150
)
plt.close()

# 11. Save best model pipeline
joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("\nBest model saved successfully!")
print("Training completed!")
