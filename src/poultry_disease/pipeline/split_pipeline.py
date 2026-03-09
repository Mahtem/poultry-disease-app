import pandas as pd
from sklearn.model_selection import train_test_split

# Load original labels
df = pd.read_csv("artifacts/data_ingestion/labels.csv")

# First split: Train (70%) and Temp (30%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: Validation (15%) and Test (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Save new CSVs
train_df.to_csv("artifacts/data_ingestion/train.csv", index=False)
val_df.to_csv("artifacts/data_ingestion/val.csv", index=False)
test_df.to_csv("artifacts/data_ingestion/test.csv", index=False)

print("Splitting done!")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df)) 