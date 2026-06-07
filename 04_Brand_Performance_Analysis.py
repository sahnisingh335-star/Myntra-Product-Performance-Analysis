import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("myntra_cleaned.csv")

# Top 10 brands by product count
top_brands = df["brand_name"].value_counts().head(10)

print(top_brands)

# Plot
plt.figure(figsize=(12,6))
top_brands.plot(kind="bar")

plt.title("Top 10 Brands by Product Count")
plt.xlabel("Brand")
plt.ylabel("Number of Products")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()