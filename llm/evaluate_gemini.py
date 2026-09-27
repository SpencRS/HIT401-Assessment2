import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# Load Gemini LLM results
df = pd.read_csv("data/gemini_llm_results.csv")

# Remove records where the API call did not produce a valid prediction
valid_results = df.dropna(subset=["predicted_label"]).copy()

actual = valid_results["actual_label"].astype(int)
predicted = valid_results["predicted_label"].astype(int)

accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)
cm = confusion_matrix(actual, predicted)

print("Gemini LLM Evaluation")
print("---------------------")
print(f"Valid predictions: {len(valid_results)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predicted,
        target_names=["Non-MHMisinfo", "MHMisinfo"],
        zero_division=0,
    )
)
