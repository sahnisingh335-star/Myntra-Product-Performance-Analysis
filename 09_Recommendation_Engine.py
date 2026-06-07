import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("myntra_feature_engineered.csv")

# Combine text
df["combined"] = (
    df["brand_name"].astype(str) + " " +
    df["pants_description"].astype(str)
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Example product index
product_index = 0

# Similarity only for one product
similarities = cosine_similarity(
    tfidf_matrix[product_index],
    tfidf_matrix
).flatten()

# Top 5 similar products
similar_indices = similarities.argsort()[-6:-1][::-1]

print("\nSelected Product:")
print(df.iloc[product_index]["brand_name"],
      "-",
      df.iloc[product_index]["pants_description"])

print("\nRecommended Products:\n")

for idx in similar_indices:
    print(
        df.iloc[idx]["brand_name"],
        "-",
        df.iloc[idx]["pants_description"]
    )