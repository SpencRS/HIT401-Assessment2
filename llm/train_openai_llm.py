import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split


# Load the API key from the .env file
load_dotenv()
# Create the OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




# Load the prepped dataset
df = pd.read_csv("data/prepared_mhm_dataset.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# Split the dataset using the same method as the traditional ML models
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTest class distribution:")
print(test_df["label"].value_counts())

# Display the first test record
first_test_record = test_df.iloc[0]

print("\nFirst test record:")
print("Text:")
print(first_test_record["text"])
print("\nActual label:", first_test_record["label"])


# Run the LLM on the full test set

results = []

for index, (_, row) in enumerate(test_df.iterrows(), start=1):

    prompt = f"""
You are classifying social media content for mental health misinformation.

Classify the following content into exactly one of these categories:

MHMISINFO
NON-MHMISINFO

MHMISINFO means the content contains or promotes misleading, false,
unsupported, or potentially harmful claims about mental health,
mental health conditions, or their treatment.

NON-MHMISINFO means the content does not contain such misinformation.

Content:
{row["text"]}
"""

    try:

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "mental_health_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classification": {
                                "type": "string",
                                "enum": ["MHMISINFO", "NON-MHMISINFO"]
                            }
                        },
                        "required": ["classification"],
                        "additionalProperties": False
                    }
                }
            }
        )

        prediction_data = json.loads(response.output_text)

        prediction_text = prediction_data["classification"]

        if prediction_text == "MHMISINFO":
            prediction = 1
        else:
            prediction = 0

        results.append({
            "record": index,
            "actual_label": row["label"],
            "predicted_label": prediction,
            "prediction_text": prediction_text,
            "error": ""
        })

        print(
            f"Record {index}/{len(test_df)} | "
            f"Prediction: {prediction_text} | "
            f"Actual: {row['label']}"
        )

    except Exception as e:

        results.append({
            "record": index,
            "actual_label": row["label"],
            "predicted_label": None,
            "prediction_text": None,
            "error": str(e)
        })

        print(
            f"Record {index}/{len(test_df)} | "
            f"ERROR: {e}"
        )


# Save the LLM predictions

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/openai_llm_results.csv",
    index=False
)

print("\nLLM predictions saved to data/openai_llm_results.csv")