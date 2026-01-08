import joblib
import pandas as pd
import re

# Load models
vec = joblib.load("models/vectorizer.pkl")
cls_model = joblib.load("models/classifier.pkl")
reg_model = joblib.load("models/regressor.pkl")

# Technical Rewrite
desc = "Given an integer w representing the weight of a watermelon. Determine if it can be divided into two parts, each weighing an even number of kilos."
desc_clean = "Given an integer w. Print YES if w is even and greater than 2, otherwise print NO."

def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)

# Test with ONLY description to see if that explains the higher score
combined = clean_text(desc)
combined_clean = clean_text(desc_clean)
print(f"Combined text length: {len(combined)}")

# Predict Original
v = vec.transform([combined])
c_pred = cls_model.predict(v)[0]
r_pred = reg_model.predict(v)[0]

# Predict Clean
v_clean = vec.transform([combined_clean])
c_pred_clean = cls_model.predict(v_clean)[0]
r_pred_clean = reg_model.predict(v_clean)[0]

label_reverse = {0: "Easy", 1: "Medium", 2: "Hard"}

print(f"Original Prediction: {label_reverse[c_pred]} (Score: {r_pred:.2f})")
print(f"Modified Prediction: {label_reverse[c_pred_clean]} (Score: {r_pred_clean:.2f})")
