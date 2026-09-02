#imported libs here
import pandas as pd

#reads the csv file into a variable df
df = pd.read_csv("data/videos_MHMisinfo_Gold.csv")

#storing a copy of the fields we need into the data variable dataframe
data = df[["video_title", "audio_transcript", "label"]].copy()

#replace missing audio transcripts with empty strings
data["audio_transcript"] = data["audio_transcript"].fillna("")

#ensures that the missing titles are also filled with empty strings, and converts the values to python strings
data["video_title"] = data["video_title"].fillna("").astype(str)
data["audio_transcript"] = data["audio_transcript"].astype(str)

#Combine the video title and transcript fields into one field called text
data["text"] = (
    data["video_title"] + " " + data["audio_transcript"]
)

# converting the text to lowercase to ensure consistency
data["text"] = data["text"].str.lower()

# Remove unnecessary whitespace
data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Convert the labels:
# 0  = non-MHMisinfo
# -1 = MHMisinfo
#convert -1 to 1 for easier binary classification.
data["label"] = data["label"].replace({-1: 1})

# Keep only the columns required for modelling
data = data[["text", "label"]]

# Display the prepared data
print("Prepared dataset shape:", data.shape)

print("\nClass distribution:")
print(data["label"].value_counts())

print("\nFirst prepared example:")
print(data.iloc[0]["text"])

print("\nFirst prepared label:")
print(data.iloc[0]["label"])

# Save the prepared dataset
data.to_csv("data/prepared_mhm_dataset.csv", index=False)

print("\nPrepared dataset saved to:")
print("data/prepared_mhm_dataset.csv")