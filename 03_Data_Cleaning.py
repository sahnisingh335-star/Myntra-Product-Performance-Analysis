import pandas as pd

df = pd.read_csv("myntra_dataset_ByScraping.csv")

# Remove duplicates
df = df.drop_duplicates()

# Standardize discount format
df["discount_percent"] = df["discount_percent"].apply(
    lambda x: x * 100 if x <= 1 else x
)

print("Cleaned Shape:", df.shape)

print("\nDiscount Summary:")
print(df["discount_percent"].describe())

# Save cleaned dataset
df.to_csv("myntra_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")