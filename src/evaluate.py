import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from preprocess import load_and_preprocess

# ---------------- LOAD DATA ----------------
print("⏳ Loading and preprocessing data...")
df = load_and_preprocess("data/problems.csv")

label_map = {"Easy": 0, "Medium": 1, "Hard": 2}
label_reverse = {0: "Easy", 1: "Medium", 2: "Hard"}
df["label"] = df["problem_class"].map(label_map)

X = df["combined_text"]
y_cls = df["label"]           # For Classification
y_reg = df["problem_score"]   # For Regression

# ---------------- VECTORIZATION ----------------
print("⏳ Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_vec = vectorizer.fit_transform(X)

# ---------------- MODELS ----------------
classifier = LinearSVC(
    class_weight="balanced",
    C=1.0,
    max_iter=5000
)

regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# ---------------- 5-FOLD SPLIT ----------------
# We will use the same fold (Fold 3) for both to be consistent with previous runs
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🚀 STARTING EVALUATION (FOLD 3)...\n")

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_vec, y_cls), start=1):
    if fold_idx == 3:
        # Split Data
        X_train, X_test = X_vec[train_idx], X_vec[test_idx]
        
        # Classification Targets
        y_cls_train, y_cls_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
        
        # Regression Targets
        y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        
        # --- CLASSIFICATION EVALUATION ---
        classifier.fit(X_train, y_cls_train)
        y_cls_pred = classifier.predict(X_test)
        
        acc = accuracy_score(y_cls_test, y_cls_pred)
        cm = confusion_matrix(y_cls_test, y_cls_pred)
        
        print("📊 CLASSIFICATION RESULTS")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix (Rows=True, Cols=Pred):")
        print(f"{'':<10} {'Easy':<8} {'Medium':<8} {'Hard':<8}")
        for i, label in label_reverse.items():
            row_str = f"{label:<10} "
            for j in range(3):
                row_str += f"{cm[i, j]:<8} "
            print(row_str)

        # --- REGRESSION EVALUATION ---
        regressor.fit(X_train, y_reg_train)
        y_reg_pred = regressor.predict(X_test)
        
        # Clip predictions to valid range [0, 10]? 
        # The prompt implies we should just predict, but clipping makes sense for interpretation.
        # We'll calculate metrics on raw output to be honest about model performance, 
        # but note that the app clips it.
        
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        
        print("\n📈 REGRESSION RESULTS")
        print(f"MAE  (Mean Absolute Error): {mae:.4f}")
        print(f"RMSE (Root Mean Sq. Error): {rmse:.4f}")
        
        break

print("\n✅ Evaluation complete.")
