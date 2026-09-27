import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD GEMINI API KEY
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Add GEMINI_API_KEY=your_key to a local .env file."
    )

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.1-flash-lite"


# ============================================================
# 2. LOAD PREPARED DATASET
# ============================================================

df = pd.read_csv("data/prepared_mhm_dataset.csv")

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# 3. SAME TRAIN/TEST SPLIT AS OTHER EXPERIMENTS
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTest class distribution:")
print(test_df["label"].value_counts())


# ============================================================
# 4. SHOW FIRST TEST RECORD
# ============================================================

print("\nFirst test record:")
print("Text:")
print(test_df.iloc[0]["text"])

print("\nActual label:", test_df.iloc[0]["label"])


# ============================================================
# 5. GEMINI CLASSIFICATION FUNCTION
# ============================================================

def classify_with_gemini(text):

    prompt = f"""
You are classifying social media content for mental health misinformation.

Classify the following content into exactly one of these categories:

MHMISINFO
NON-MHMISINFO

MHMISINFO means the content contains or promotes misleading, false,
unsupported, or potentially harmful claims about mental health,
mental health conditions, or their treatment.

NON-MHMISINFO means the content does not contain such misinformation.

Return ONLY one label: MHMISINFO or NON-MHMISINFO.

Content:
{text}
"""

    # Maximum number of attempts for temporary API failures
    max_retries = 6

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )

            prediction_text = response.text.strip().upper()

            # Exact response
            if prediction_text == "MHMISINFO":
                return "MHMISINFO", 1

            if prediction_text == "NON-MHMISINFO":
                return "NON-MHMISINFO", 0

            # Handle occasional extra text
            if "NON-MHMISINFO" in prediction_text:
                return "NON-MHMISINFO", 0

            if "MHMISINFO" in prediction_text:
                return "MHMISINFO", 1

            raise ValueError(
                f"Unexpected Gemini response: {response.text!r}"
            )

        except Exception as exc:

            error_message = str(exc)

            # ------------------------------------------------
            # Retry temporary Gemini errors
            # ------------------------------------------------

            temporary_error = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
                or "503" in error_message
                or "UNAVAILABLE" in error_message
            )

            if temporary_error:

                # Increasing wait time between retries
                wait_time = 30 * (attempt + 1)

                print(
                    f"  Gemini temporarily unavailable/rate limited. "
                    f"Waiting {wait_time} seconds before retry "
                    f"({attempt + 1}/{max_retries})..."
                )

                time.sleep(wait_time)

                continue

            # ------------------------------------------------
            # Do not retry unexpected errors
            # ------------------------------------------------

            raise

    # All retry attempts failed
    raise RuntimeError(
        "Gemini request failed after maximum retry attempts."
    )


# ============================================================
# 6. RUN GEMINI ON TEST SET
# ============================================================

results = []

for index, (_, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    prediction_text = None
    prediction = None
    error = ""

    try:

        prediction_text, prediction = classify_with_gemini(
            row["text"]
        )

    except Exception as exc:

        error = str(exc)

    # Save result
    results.append(
        {
            "record": index,
            "actual_label": int(row["label"]),
            "predicted_label": prediction,
            "prediction_text": prediction_text,
            "error": error,
        }
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    if error:

        print(
            f"Record {index}/{len(test_df)} | "
            f"ERROR: {error}"
        )

    else:

        print(
            f"Record {index}/{len(test_df)} | "
            f"Prediction: {prediction_text} | "
            f"Actual: {row['label']}"
        )

    # --------------------------------------------------------
    # Pause between normal requests
    # --------------------------------------------------------

    time.sleep(5)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/gemini_llm_results.csv",
    index=False
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n-----------------------------")
print("Gemini experiment completed")
print("-----------------------------")

print(
    f"Total test records: "
    f"{len(results_df)}"
)

print(
    f"Valid predictions: "
    f"{results_df['predicted_label'].notna().sum()}"
)

print(
    f"Failed predictions: "
    f"{results_df['predicted_label'].isna().sum()}"
)

print("\nGemini predictions saved to:")
print("data/gemini_llm_results.csv")