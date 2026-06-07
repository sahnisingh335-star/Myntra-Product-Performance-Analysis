import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("myntra_feature_engineered.csv")

# Features
features = [
    "price",
    "discount_percent",
    "ratings",
    "Popularity_Score"
]

X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Summary
summary = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\nCLUSTER SUMMARY\n")
print(summary)

# Save
df.to_csv(
    "myntra_segmented.csv",
    index=False
)

print("\nSegmentation Completed!")