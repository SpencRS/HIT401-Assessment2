# HIT401 - Mental Health Misinformation Classification

## 1. Project Overview

The project investigates the use of machine-learning techniques to classify
mental-health misinformation in social-media video content. The technical implementation
uses the MentalMisinfo dataset and compares three TF-IDF based classification approaches
with zero-shot large language model classification approaches using OpenAI and Gemini.

The current implementation focuses on text classification. Real-time social-media
monitoring and the proposed visual analytics dashboard are outside the current
implementation and remain future components of the project.

## 2. Environment Requirements

- Python 3.13.15
- pandas 3.0.5
- scikit-learn 1.9.0
- openai 3.16.2
- google-genai 1.55.0
- python-dotenv 1.2.3

The programs were developed and tested using a Python Virtual Environment.

The OpenAI and Gemini experiments require API keys. The API keys should be stored
locally in a `.env` file and should not be included in the repository.

## 3. Project Files

### Data

- `data/videos_MHMisinfo_Gold.csv`
  - Original MentalMisinfo video dataset used for this project.

- `data/prepared_mhm_dataset.csv`
  - Prepared dataset containing the combined text field and recoded labels.

- `data/openai_llm_results.csv`
  - Results produced by the OpenAI zero-shot classification experiment.

- `data/gemini_llm_results.csv`
  - Results produced by the Gemini zero-shot classification experiment.

### Source Code

- `explore_data.py`
  - Inspects the dataset structure, labels, missing values and class
    distribution.

- `prepare_data.py`
  - Prepares the dataset for machine-learning experiments by combining the
    video title and audio transcript, handling missing transcripts,
    normalising the text and converting the original misinformation label
    from -1 to 1.

- `traditional_ml/train_baseline.py`
  - Runs Experiment 1 using TF-IDF with Logistic Regression.

- `traditional_ml/train_balanced_lr.py`
  - Runs Experiment 2 using TF-IDF with class-balanced Logistic Regression.

- `traditional_ml/train_svm.py`
  - Runs Experiment 3 using TF-IDF with class-balanced Linear SVM.

- `llm/train_openai_llm.py`
  - Runs the OpenAI zero-shot classification experiment using the same
    test set as the traditional machine-learning experiments.

- `llm/evaluate_openai.py`
  - Evaluates the saved OpenAI predictions using accuracy, precision,
    recall, F1-score and a confusion matrix.

- `llm/test_openai.py`
  - Provides testing functionality for the OpenAI LLM implementation.

- `llm/train_gemini_llm.py`
  - Runs the Gemini zero-shot classification experiment using the same
    test set as the traditional machine-learning experiments.

- `llm/evaluate_gemini.py`
  - Evaluates the saved Gemini predictions using accuracy, precision,
    recall, F1-score and a confusion matrix.

- `llm/test_gemini.py`
  - Provides testing functionality for the Gemini LLM implementation.

## 4. Dataset Source

The dataset used in this project is the MentalMisinfo dataset provided by:
Nguyen et al. (2025).

Specific file used was:

`videos_MHMisinfo_Gold.csv`

The dataset was obtained from the official research repository:
https://github.com/cuongnguyenx/MHMisinfo

Associated with following research paper:

Nguyen, V.C., Jain, M., Chauhan, A., Soled, H.J., Lesmes, S.A., Li, Z.,
Birnbaum, M.L., Tang, S.X., Kumar, S. and De Choudhury, M. (2025).
Supporters and Skeptics: LLM-Based Analysis of Engagement with Mental
Health (Mis)Information Content on Video-Sharing Platforms. Proceedings
of the International AAAI Conference on Web and Social Media, 19,
pp.1329–1345. doi:10.1609/icwsm.v19i1.35875.

Dataset Repository:
https://github.com/cuongnguyenx/MHMisinfo

## 5. Dataset Preparation

Original dataset contains 739 video records.

The original dataset uses:

- 0 = non-misinformation
- -1 = misinformation

For machine-learning classification, the misinformation label was changed from -1 to 1.

Video title and audio transcript were combined into a single text field.
Missing audio transcripts were replaced with empty strings.

Video description was not used for modelling because a substantial number of records
contained missing or empty descriptions.

Overall prepared dataset contains:

- 739 total records
- 619 non-misinformation records
- 120 misinformation records

## 6. How to Run

The programs should be executed in the following order.

### Step 1 - Explore the dataset

```bash
python explore_data.py
```

This displays the dataset dimensions, columns, label distribution, missing values,
sample records and label percentages.

### Step 2 - Prepare the dataset

```bash
python prepare_data.py
```

This creates:

`data/prepared_mhm_dataset.csv`

### Step 3 - Run Experiment 1

```bash
python traditional_ml/train_baseline.py
```

This runs the baseline Logistic Regression + TF-IDF experiment.

### Step 4 - Run Experiment 2

```bash
python traditional_ml/train_balanced_lr.py
```

This runs the TF-IDF + class-balanced Logistic Regression model.

### Step 5 - Run Experiment 3

```bash
python traditional_ml/train_svm.py
```

This runs the TF-IDF + class-balanced Linear SVM model.

### Step 6 - Run the OpenAI LLM Experiment

Before running the OpenAI experiment, the OpenAI API key must be configured
locally in a `.env` file.

```bash
python llm/train_openai_llm.py
```

This sends the same test records used by the traditional machine-learning
experiments to the OpenAI model for zero-shot classification.

The results are saved to:

`data/openai_llm_results.csv`

### Step 7 - Evaluate the OpenAI Results

```bash
python llm/evaluate_openai.py
```

This calculates accuracy, precision, recall, F1-score and the confusion matrix
from the saved OpenAI predictions.

### Step 8 - Run the Gemini LLM Experiment

Before running the Gemini experiment, the Gemini API key must be configured
locally in a `.env` file.

```bash
python llm/train_gemini_llm.py
```

This sends the same test records used by the traditional machine-learning
experiments to the Gemini model for zero-shot classification.

The results are saved to:

`data/gemini_llm_results.csv`

### Step 9 - Evaluate the Gemini Results

```bash
python llm/evaluate_gemini.py
```

This calculates accuracy, precision, recall, F1-score and the confusion matrix
from the saved Gemini predictions.

## 7. Experimental Setup

All three traditional machine-learning experiments use the same prepared dataset
and an 80/20 stratified train-test split.

The OpenAI and Gemini experiments use the same 148-record test set so that their
predictions can be compared against the traditional machine-learning experiments.

### Training Data

- 591 total records
- 495 non-misinformation
- 96 misinformation

### Testing Data

- 148 total records
- 124 non-misinformation
- 24 misinformation

A `random_state` value of 42 was used so that the same train-test split can be
reproduced across the experiments.

The traditional machine-learning models are trained using the training portion
of the dataset. The OpenAI and Gemini experiments are zero-shot classification
approaches and do not use the training labels as examples.

## 8. Traditional Machine-Learning Results

The three traditional machine-learning experiments produced the following results
on the 148-record test set.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 83.78% | 0.00% | 0.00% | 0.00% |
| TF-IDF + Balanced Logistic Regression | 85.81% | 54.84% | 70.83% | 61.82% |
| TF-IDF + Balanced Linear SVM | 85.81% | 57.89% | 45.83% | 51.16% |

The baseline Logistic Regression model predicted every test record as
non-misinformation. Its accuracy therefore reflects the majority class in the
test set rather than successful identification of misinformation.

Class balancing improved the ability of the Logistic Regression and Linear SVM
models to identify misinformation.

## 9. OpenAI LLM Experiment

The OpenAI experiment used a zero-shot classification approach. The model was
given the combined video title and audio transcript and was asked to classify
each record as either `MHMISINFO` or `NON-MHMISINFO`.

The ground-truth labels were not provided to the model.

The same 148-record test set used for the traditional machine-learning
experiments was used for the OpenAI experiment. Structured JSON output was used
so that the classification could be converted into the same numeric labels
used by the dataset.

The experiment produced 148 valid predictions with no recorded API errors.

### OpenAI Results

| Metric | Result |
|---|---:|
| Accuracy | 76.35% |
| Precision | 40.35% |
| Recall | 95.83% |
| F1-score | 56.79% |

Confusion matrix:

```text
[[90, 34],
 [ 1, 23]]
```

The OpenAI model correctly identified 23 of the 24 misinformation records,
resulting in a recall of 95.83%. However, 34 non-misinformation records were
also classified as misinformation, resulting in a precision of 40.35%.

This shows a different precision-recall balance compared with the traditional
machine-learning models. The OpenAI model produced a higher recall than the
three traditional approaches, while the balanced Logistic Regression model
produced the highest F1-score among the experiments completed so far.

The comparison is based on this particular 148-record test set and should be
treated as preliminary rather than as a general measure of model performance.

## 10. Gemini LLM Experiment

A Gemini LLM experiment was conducted to classify mental-health misinformation
using the same prepared dataset and test split as the other experiments.

The experiment used:

- **Model:** `gemini-3.1-flash-lite`
- **Task:** Binary classification of mental-health misinformation
- **Classes:** `MHMISINFO` and `NON-MHMISINFO`
- **Test records:** 148
- **Valid predictions:** 148
- **Failed predictions:** 0

The Gemini model was given the text content and instructed to return only one
of the two classification labels. The ground-truth labels were not provided
to the model.

The predictions were saved to:

`data/gemini_llm_results.csv`

### Gemini Results

| Metric | Score |
|---|---:|
| Accuracy | 78.38% |
| Precision | 42.59% |
| Recall | 95.83% |
| F1-score | 58.97% |

Confusion matrix:

```text
[[93, 31],
 [ 1, 23]]
```

The results show that Gemini correctly identified 23 out of the 24 mental-health
misinformation cases, resulting in a recall of 95.83%. However, its precision
was lower because 31 non-misinformation records were incorrectly classified as
misinformation.

This indicates that the Gemini model was effective at detecting potential
misinformation but was relatively aggressive in classifying content as
misinformation. The Gemini results can therefore be compared with the
traditional machine-learning and OpenAI experiments using the same test set.

The Gemini experiment implementation is available in:

- `llm/train_gemini_llm.py`
- `llm/evaluate_gemini.py`
- `llm/test_gemini.py`

The generated results are available in:

`data/gemini_llm_results.csv`

## 11. LLM Comparison

Both LLM experiments used zero-shot classification and the same 148-record
test set. This allows their results to be compared using the same ground-truth
labels.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| OpenAI | 76.35% | 40.35% | 95.83% | 56.79% |
| Gemini | 78.38% | 42.59% | 95.83% | 58.97% |

Both LLM experiments identified 23 of the 24 misinformation records, resulting
in the same recall of 95.83%.

The Gemini experiment produced fewer false-positive classifications than the
OpenAI experiment, with 31 non-misinformation records incorrectly classified as
misinformation compared with 34 for OpenAI.

The results are based on the same 148-record test set and should therefore be
interpreted as an experimental comparison rather than a general measure of
LLM performance.

## 12. Limitations

The experiments were conducted using a relatively small dataset containing
739 records, with only 120 records labelled as misinformation.

The test set contained 24 misinformation records. Therefore, individual
classification errors can have a noticeable effect on the reported metrics.

The LLM experiments were conducted using zero-shot classification and did not
provide the ground-truth labels to the models.

The results are based on a single stratified 80/20 train-test split and a
single test set of 148 records. The results should therefore be considered
preliminary and may not generalise to other datasets or social-media content.

Real-time social-media monitoring and the proposed visual analytics dashboard
were outside the current implementation scope.

## 13. References

Nguyen, V.C., Jain, M., Chauhan, A., Soled, H.J., Lesmes, S.A., Li, Z.,
Birnbaum, M.L., Tang, S.X., Kumar, S. and De Choudhury, M. (2025).
Supporters and Skeptics: LLM-Based Analysis of Engagement with Mental
Health (Mis)Information Content on Video-Sharing Platforms. Proceedings
of the International AAAI Conference on Web and Social Media, 19,
pp.1329–1345. doi:10.1609/icwsm.v19i1.35875.
