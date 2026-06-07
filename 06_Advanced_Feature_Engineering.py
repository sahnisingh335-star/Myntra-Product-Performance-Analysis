import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("myntra_cleaned.csv")

# Popularity Score
df["Popularity_Score"] = (
    df["ratings"] *
    df["number_of_ratings"]
)

# Value Score
df["Value_Score"] = (
    df["ratings"] *
    df["number_of_ratings"] *
    df["discount_percent"]
) / df["price"]

# Revenue Potential
df["Revenue_Potential"] = (
    df["price"] *
    df["number_of_ratings"]
)

# Customer Trust Score
df["Customer_Trust_Score"] = (
    df["ratings"] *
    np.log1p(df["number_of_ratings"])
)

print("\nTop 10 Products by Popularity Score:\n")

print(
    df.nlargest(
        10,
        "Popularity_Score"
    )[
        [
            "brand_name",
            "pants_description",
            "Popularity_Score"
        ]
    ]
)

# Save new dataset
df.to_csv(
    "myntra_feature_engineered.csv",
    index=False
)

print("\nFeature engineered dataset saved successfully!")