import pandas as pd

df = pd.read_csv("myntra_dataset_ByScraping.csv")

print("Original Shape:", df.shape)

# Duplicate count
print("Duplicates:", df.duplicated().sum())

# Check discount values > 1
print("\nDiscount > 1")
print(df[df["discount_percent"] > 1].head())

print("\nRows with discount > 1:")
print((df["discount_percent"] > 1).sum())