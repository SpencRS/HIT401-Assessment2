import pandas as pd


df = pd.read_csv("data/videos_MHMisinfo_Gold.csv")

# show dataset size
print("Dataset shape:", df.shape)

# display the columns/fields
print("\nColumns:")
print(df.columns.tolist())

# show the original distribution for the labels
print("\nLabel distribution:")
print(df["label"].value_counts())

# display the missing values
print("\nMissing values:")
print(df.isnull().sum())

# show the first five records in the table
print("\nFirst 5 records:")
print(df.head())

# percentage display for the labels
print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)