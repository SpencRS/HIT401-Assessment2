import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Load the OpenAI LLM results
df = pd.read_csv("data/openai_llm_results.csv")

# Remove records that failed
valid_results = df.dropna(subset=["predicted_label"])

actual = valid_results["actual_label"]
predicted = valid_results["predicted_label"]

# Calculate evaluation metrics
accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)

# Calculate confusion matrix
cm = confusion_matrix(actual, predicted)

print("OpenAI LLM Evaluation")
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
        zero_division=0
    )
)