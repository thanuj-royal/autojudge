import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import load_and_preprocess

os.makedirs("models", exist_ok=True)

df = load_and_preprocess("data/problems.csv")

label_map = {"Easy": 0, "Medium": 1, "Hard": 2}
df["label"] = df["problem_class"].map(label_map)

X = df["combined_text"]
y = df["label"]

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

classifier = LinearSVC(
    class_weight="balanced",
    C=1.0,
    max_iter=5000
)

classifier.fit(X_vec, y)

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(classifier, "models/classifier.pkl")

print("✅ Final classifier trained on full dataset")
