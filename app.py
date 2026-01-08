import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG & STYLING -----------------
st.set_page_config(page_title="AutoJudge - AI Problem Difficulty Predictor", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f5f7f9;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #4CAF50; color: white;}
    .prediction-box {padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;}
    .metric-card {text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #fff;}
    .tag {background-color: #e0e0e0; color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-right: 5px;}
</style>
""", unsafe_allow_html=True)

# ----------------- LOADING MODELS -----------------
@st.cache_resource
def load_models():
    vec = joblib.load("models/vectorizer.pkl")
    cls_model = joblib.load("models/classifier.pkl")
    reg_model = joblib.load("models/regressor.pkl")
    
    # Load Recommender Artifacts (Graceful fallback if not present)
    try:
        rec_vec = joblib.load("models/recommender_vectorizer.pkl")
        rec_mat = joblib.load("models/recommender_matrix.pkl")
        rec_meta = joblib.load("models/problem_metadata.pkl")
    except Exception as e:
        rec_vec, rec_mat, rec_meta = None, None, None
        
    return vec, cls_model, reg_model, rec_vec, rec_mat, rec_meta

vectorizer, classifier, regressor, rec_vectorizer, rec_matrix, df_metadata = load_models()

label_reverse = {0: "Easy", 1: "Medium", 2: "Hard"}

# ----------------- UTILS -----------------
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9\s]", " ", text)

def get_predicted_tags(text):
    text = text.lower()
    tags = []
    keywords = {
        "Graph": ["graph", "node", "edge", "vertex", "dfs", "bfs", "tree", "connected component"],
        "Dynamic Programming": ["dp", "dynamic programming", "optimal substructure", "memoization", "maximize", "minimize"],
        "Math": ["prime", "modulo", "gcd", "lcm", "integer", "divisible", "math", "number theory"],
        "Geometry": ["point", "line", "circle", "convex hull", "polygon", "coordinate", "area"],
        "Strings": ["substring", "palindrome", "suffix", "prefix", "string", "character"],
        "Data Structures": ["stack", "queue", "heap", "priority queue", "linked list", "array"],
        "Greedy": ["greedy", "sort", "optimal", "choose best"]
    }
    
    for tag, words in keywords.items():
        if any(w in text for w in words):
            tags.append(tag)
            
    return tags[:3] # Return top 3 detected tags

def recommend_similar_problems(input_text):
    if rec_vectorizer is None or rec_matrix is None:
        return []
    
    # Vectorize input
    input_vec = rec_vectorizer.transform([clean_text(input_text)])
    
    # Calculate Cosine Similarity
    scores = cosine_similarity(input_vec, rec_matrix)[0]
    
    # Get Top Indices
    top_indices = scores.argsort()[-6:][::-1] # Top 6
    
    recommendations = []
    for idx in top_indices:
        # Filter out if score is too low
        if scores[idx] < 0.1: continue
        
        item = df_metadata.iloc[idx]
        recommendations.append({
            "title": item['title'],
            "url": item['url'],
            "class": item['problem_class'],
            "score": item['problem_score'],
            "similarity": scores[idx]
        })
        
    return recommendations[:5]

# ----------------- UI LAYOUT -----------------
col1, col2 = st.columns([3, 1])

with col1:
    st.title("AutoJudge ⚖️")
    st.write("AI-Powered Problem Difficulty Predictor")

    with st.expander("📝 Problem Details", expanded=True):
        desc = st.text_area("Problem Description", height=150, placeholder="")
        c1, c2 = st.columns(2)
        inp = c1.text_area("Input Description", height=100, placeholder="e.g. First line contains N...")
        out = c2.text_area("Output Description", height=100, placeholder="e.g. Print the maximum value...")

    if st.button("Predict Difficulty"):
        if not desc:
            st.warning("Please enter at least a problem description.")
        else:
            # PREDICTION
            combined = clean_text(desc + " " + inp + " " + out)
            vec = vectorizer.transform([combined])
            cls_pred = classifier.predict(vec)[0]
            raw_score = float(regressor.predict(vec)[0])
            
            difficulty = label_reverse[cls_pred]
            
            # ----------------- CALIBRATION UTILS -----------------
            # Removed artificial forcing of score based on class.
            # Letting the models speak for themselves.
            
            final_score = min(10.0, max(0.0, raw_score)) # Cap at [0, 10]
            
            score = round(final_score, 2)
            
            # Tags
            tags = get_predicted_tags(combined)
            
            # ----------------- RESULTS -----------------
            st.markdown("### 🔍 Analysis Results")
            
            # Difficulty & Score
            r1, r2, r3 = st.columns(3)
            
            color_map = {"Easy": "#4CAF50", "Medium": "#ff9800", "Hard": "#f44336"}
            diff_color = color_map.get(difficulty, "#333")
            
            with r1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {diff_color}; margin:0;">{difficulty}</h3>
                    <p style="margin:0; color:#666;">Predicted Class</p>
                </div>
                """, unsafe_allow_html=True)
                
            with r2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #333; margin:0;">{score:.2f} / 10</h3>
                    <p style="margin:0; color:#666;">Difficulty Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tags Display
            with r3:
                st.markdown("##### 🏷️ Predicted Tags")
                if tags:
                    tag_html = "".join([f'<span class="tag">{t}</span>' for t in tags])
                    st.markdown(tag_html, unsafe_allow_html=True)
                else:
                    st.write("No specific tags detected.")
            
            st.progress(score / 10.0)
            
            # ----------------- RECOMMENDATIONS (SIDEBAR TRIGGER) -----------------
            recs = recommend_similar_problems(combined)
            if recs:
                st.session_state['recs'] = recs

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("📚 Similar Problems")
    st.write("Based on your input, here are similar problems from our database:")
    
    if 'recs' in st.session_state:
        for r in st.session_state['recs']:
            with st.container():
                st.markdown(f"**[{r['title']}]({r['url']})**")
                st.caption(f"Class: {r['class']} | Score: {r['score']} | Match: {int(r['similarity']*100)}%")
                st.markdown("---")
    else:
        st.info("Run a prediction to see recommendations here.")
