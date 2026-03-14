import pandas as pd

# ===== LOAD DATASET =====
file_path = r"C:\Users\punit\OneDrive\Documents\Fake-news_detection\datasets\multimodal_train.tsv.zip"

try:
    data = pd.read_csv(
        file_path,
        sep="\t",
        compression="zip",
        low_memory=False
    )
    print("Dataset loaded successfully ✅")
except Exception as e:
    print("Error loading dataset ❌:", e)
    exit()

# ===== SHOW COLUMNS =====
print("\nAll Columns:")
print(list(data.columns))

# ===== SET YOUR COLUMN NAMES =====
TEXT_COLUMN = "clean_title"
LABEL_COLUMN = "2_way_label"

# ===== VALIDATE COLUMNS =====
if TEXT_COLUMN not in data.columns or LABEL_COLUMN not in data.columns:
    print("\n❌ Column names incorrect. Check above column list.")
    exit()

# ===== KEEP REQUIRED COLUMNS =====
data = data[[TEXT_COLUMN, LABEL_COLUMN]]

# ===== DROP MISSING VALUES =====
data = data.dropna()

# Optional: Remove empty strings
data = data[data[TEXT_COLUMN].str.strip() != ""]

# ===== RENAME COLUMNS (Recommended for BERT Training) =====
data.columns = ["text", "label"]

print("\nAfter Cleaning:")
print(data.head())

print("\nDataset shape:", data.shape)

# ===== SAVE CLEANED FILE =====
data.to_csv("cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully! 🚀")