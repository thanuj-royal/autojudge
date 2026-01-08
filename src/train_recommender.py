import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import load_and_preprocess

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("⏳ Loading and preprocessing data for Recommender...")
df = load_and_preprocess("data/problems.csv")

# We need the metadata to link back to the actual problem details
# We need the metadata to link back to the actual problem details
# Check if 'url' exists, otherwise provide a default
cols_to_keep = ["title", "problem_class", "problem_score"]
if "url" in df.columns:
    cols_to_keep.append("url")
else:
    df["url"] = "#" # Default placeholder
    cols_to_keep.append("url")

metadata = df[cols_to_keep].copy()

print("⏳ Vectorizing text for Recommender...")
# We use a separate vectorizer or the same one. To be safe and self-contained, 
# we'll fit a new one on the full dataset specifically for similarity search.
vectorizer = TfidfVectorizer(
    max_features=15000, # Slightly more features for better differentiation
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(df["combined_text"])

print(f"✅ Created TF-IDF matrix with shape: {X_vec.shape}")

# Save Artifacts
print("💾 Saving artifacts to models/...")
joblib.dump(vectorizer, "models/recommender_vectorizer.pkl")
joblib.dump(X_vec, "models/recommender_matrix.pkl")
joblib.dump(metadata, "models/problem_metadata.pkl")

print("🚀 Recommender training complete!")
