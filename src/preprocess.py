import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    # Remove LaTeX commands (e.g., \frac, \times, \leq)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Remove missing labels
    df = df.dropna(subset=["problem_class", "problem_score"])

    # Combine text fields
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["input_description"].fillna("") + " " +
        df["output_description"].fillna("")
    )

    df["combined_text"] = df["combined_text"].apply(clean_text)

    # Normalize labels
    df["problem_class"] = df["problem_class"].str.capitalize()

    return df
