import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_and_preprocess

# 1. Load Data
print("⏳ Loading data for Regressor training...")
df = load_and_preprocess("data/problems.csv")

# 2. Vectorize
# Load the SAME vectorizer used for classification (consistent features)
vectorizer = joblib.load("models/vectorizer.pkl")
X_vec = vectorizer.transform(df["combined_text"])
y = df["problem_score"]

# 3. Train Regressor
# Switching to Random Forest for better non-linear pattern capture
print("⏳ Training Random Forest Regressor (this may take a moment)...")
regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
regressor.fit(X_vec, y)

# 4. Save
joblib.dump(regressor, "models/regressor.pkl")
print("✅ Random Forest Regressor saved to models/regressor.pkl")
