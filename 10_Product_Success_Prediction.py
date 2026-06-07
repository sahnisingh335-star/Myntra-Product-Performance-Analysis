import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("myntra_feature_engineered.csv")

# Create Target Variable
df["Popularity_Level"] = pd.qcut(
    df["Popularity_Score"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# Features
X = df[
    [
        "price",
        "discount_percent",
        "ratings",
        "number_of_ratings"
    ]
]

# Target
y = df["Popularity_Level"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("=" * 50)
print("MODEL ACCURACY")
print("=" * 50)

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

# Classification Report
print("\n")
print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Feature Importance
importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

print(importance)

# Save model predictions
df["Predicted_Popularity"] = model.predict(X)

df.to_csv(
    "myntra_ml_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")