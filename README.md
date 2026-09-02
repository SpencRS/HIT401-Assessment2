README


#HIT401 - Mental Health Misinformation Classification

## 1. Project Overview

The project investigates the use of machine-learning techniques to classify
mental-health misinformation in social-media video content. The technical implementation
uses the MentalMisinfo dataset and compares three TF-IDF based classification approaches

## 2. Environment Requirements

- Python 3.13.15
- pandas 3.0.5
- scikit-learn 1.9.0

The programs were developed and tested using a Python Virtual Environment.

## 3. Project Files

### Data

- `data/videos_MHMisinfo_Gold.csv`
  - Original MentalMisinfo video dataset was used for this project.
- `data/prepared_mhm_dataset.csv`
  - Prepared Dataset containing the combined text field and recoded labels


### Source Code

- `explore_data.py`
  - Inspects the dataset structure, labels, missing values and class
    distribution.

- `prepare_data.py`
  - Prepares the dataset for machine-learning experiments by combining the
    video title and audio transcript, handling missing transcripts,
    normalising the text and converting the original misinformation label
    from -1 to 1.

- `train_baseline.py`
  - Runs Experiment 1 using TF-IDF with Logistic Regression.

- `train_balanced_lr.py`
  - Runs Experiment 2 using TF-IDF with class-balanced Logistic Regression.

- `train_svm.py`
  - Runs Experiment 3 using TF-IDF with class-balanced Linear SVM.

## 4 - Dataset Source

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
No link as of yet -- REMINDER TO FILL THIS IF IT NEEDS TO BE ON GITHUB


## 5. Dataset Preparation

Original dataset contains 739 video records
Original labels are:
- 0 = non-misinformation
- 1 = misinformation

For machine learning classification, the misinformation label was changed from -1 to 1.

Video title and audio transcript were combined into a single text field. missing audio transcripts were replaced with empty strings.

Video description not used for modelling because a substantial number of records
contained missing or empty descriptions

Overall prepared dataset contains:
- 739 total records
- 619 non-misinformation records
- 120 misinformation records


## 6. How to run

The programs should be executed in the following order:


### Step 1 - Explore the dataset

python explore_data.py

This displays the dataset dimensions, columns, label distribution, missing values, sample records
and label percentages.

### Step 2 - Prepare the dataset


python prepare_data.py

This creates:
`data/prepared_mhm_dataset.csv`


## Step 3 - Run Experiment 1

python train_model.py

This runs the baseline Logistic Regression + TF-IDF


## Step 4 - Run Experiment 2

python train_balanced_lr.py

This runs the TF-IDF + Class-balanced logistic regression model


## Step 5 - Run Experiment 3

python train_svm.py

This runs the TF-IDF + Class-balanced Linear SVM Model.


## 7. Experimental Setup

All three experiments use the same prepared dataset and an 80/20
stratified train-test split

Training Data
- 591 total records
- 495 non-misinformation
- 96 misinformation

Testing Data
- 148 total records
- 124 non-misinformation
- 24 misinformation

A random_state value of 42 was used so that the same train-test split can be reproduced across
the experiments


## 8. Reproducibility

The supplied source code and dataset allow the experiments reported in the technical sections
of the project to be reproduced

The experiments use the same dataset, preprocessing approach, train-test
split and random state for comparison

Current implementation uses a static dataset and does not implement real-time social media
monitoring or the proposed visual analytics dashboard


## References

Nguyen, V.C., Jain, M., Chauhan, A., Soled, H.J., Lesmes, S.A., Li, Z.,
Birnbaum, M.L., Tang, S.X., Kumar, S. and De Choudhury, M. (2025).
Supporters and Skeptics: LLM-Based Analysis of Engagement with Mental
Health (Mis)Information Content on Video-Sharing Platforms. Proceedings
of the International AAAI Conference on Web and Social Media, 19,
pp.1329–1345. doi:10.1609/icwsm.v19i1.35875.
























