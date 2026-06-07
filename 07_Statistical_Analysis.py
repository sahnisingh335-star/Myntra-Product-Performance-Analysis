import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("myntra_feature_engineered.csv")

# -----------------------------
# Correlation Heatmap
# -----------------------------

plt.figure(figsize=(10,6))

sns.heatmap(
    df.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Price Distribution
# -----------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["price"],
    bins=30,
    kde=True
)

plt.title("Price Distribution")
plt.show()

# -----------------------------
# Rating Distribution
# -----------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["ratings"],
    bins=20,
    kde=True
)

plt.title("Ratings Distribution")
plt.show()

# -----------------------------
# Discount Distribution
# -----------------------------

plt.figure(figsize=(10,6))

sns.histplot(
    df["discount_percent"],
    bins=30,
    kde=True
)

plt.title("Discount Distribution")
plt.show()

print("\nStatistical Analysis Completed Successfully!")