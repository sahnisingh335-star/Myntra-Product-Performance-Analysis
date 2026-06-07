import pandas as pd

# Load cleaned dataset
df = pd.read_csv("myntra_cleaned.csv")

print("="*60)
print("DATA PROFILING REPORT")
print("="*60)

# Shape
print("\nDataset Shape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns.tolist())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Unique Values
print("\nUnique Values:")
print(df.nunique())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Top 10 Brands
print("\nTop 10 Brands:")
print(df["brand_name"].value_counts().head(10))