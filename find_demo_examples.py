import joblib
import pandas as pd
import re
from src.preprocess import clean_text

def load_models():
    # Load separate vectorizer for classifier as per training scripts
    # Note: src/train_classifier.py and src/train_regressor.py might share or reproduce vectorizer
    # Based on app.py, models/vectorizer.pkl is used.
    vec = joblib.load("models/vectorizer.pkl")
    cls_model = joblib.load("models/classifier.pkl")
    reg_model = joblib.load("models/regressor.pkl")
    return vec, cls_model, reg_model

def get_demo_candidates(csv_path="data/problems.csv"):
    df = pd.read_csv(csv_path)
    
    # We want to find examples where Pred Class == Actual Class
    vectorizer, classifier, regressor = load_models()
    
    label_reverse = {0: "Easy", 1: "Medium", 2: "Hard"}
    results = {"Easy": [], "Medium": [], "Hard": []}
    
    print("Searching for perfect demo candidates...")
    
    for idx, row in df.iterrows():
        # Clean and Combine
        desc = str(row['description']) if pd.notna(row['description']) else ""
        inp = str(row['input_description']) if pd.notna(row['input_description']) else ""
        out = str(row['output_description']) if pd.notna(row['output_description']) else ""
        
        combined = clean_text(desc + " " + inp + " " + out)
        
        # Predict
        vec_text = vectorizer.transform([combined])
        pred_idx = classifier.predict(vec_text)[0]
        pred_class = label_reverse[pred_idx]
        pred_score = float(regressor.predict(vec_text)[0])
        
        actual_class = row['problem_class']
        
        # Check match
        if pred_class == actual_class:
            # We want short, clear descriptions for the demo
            if len(desc) < 400 and len(desc) > 30: 
                print(f"\n--- FOUND {actual_class.upper()} MATCH ---")
                print(f"Title: {row.get('name', 'Unknown')}")
                print(f"Pred Score: {round(pred_score, 2)}")
                print(f"Desc: {desc}")
                print(f"Inp: {inp}")
                print(f"Out: {out}")
                
                results[actual_class].append(1)
        
        # Stop if we found enough
        if all(len(v) >= 1 for v in results.values()):
            print("\nFound one of each! Stopping.")
            break

if __name__ == "__main__":
    get_demo_candidates()
