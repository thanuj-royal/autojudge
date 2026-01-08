# Demo Video Script (2-3 Minutes)

**Requirement:** Show project explanation, approach, noise reduction details, challenges faced, and working web UI with all features.

---

## 0:00 - 0:30 : Introduction & Project Overview
**Visual:** Show the **Title Slide** or README.md on GitHub.

**Say:** 
> "Hello, I am Chakrala Reddy Thanuj Royal, ID 24114025, CSE 2nd Year. This is my project 'AutoJudge', an AI-powered system that predicts the difficulty of programming problems based solely on their text description. Manual difficulty assignment is often subjective and time-consuming. My goal was to automate this process using Machine Learning to predict both a difficulty class (Easy, Medium, Hard) and a precise numerical score from 0 to 10."

---

## 0:30 - 1:20 : Technical Approach

**Visual:** Show `src/preprocess.py` code or a slide with your approach diagram.

**Say:**
> "Let me explain my technical approach. First, I tackled **Data Noise**. Programming problems contain LaTeX mathematical formatting like backslash-frac, backslash-leq, and backslash-times. These commands are meaningless noise to a text classifier. I implemented a custom regex-based cleaning pipeline in the preprocess.py file that strips these LaTeX artifacts while preserving the actual problem content.
>
> For **Feature Extraction**, I used TF-IDF vectorization with **trigrams**. Instead of just looking at individual words, trigrams capture 3-word sequences like 'shortest path algorithm' or 'dynamic programming approach'. This gives the model much better context. I configured the vectorizer with 20,000 max features to capture a rich vocabulary.
>
> For the **models**, I used LinearSVC for classification because it's highly effective for high-dimensional sparse text data. For regression, I used RandomForestRegressor to predict the precise difficulty score. Both models share the same TF-IDF features for consistency."

---

## 1:20 - 1:50 : Advanced Features

**Visual:** Show the web app sidebar with Similar Problems.

**Say:**
> "Beyond basic prediction, I implemented two advanced features. First, **Similar Problem Recommendations**. I built a separate TF-IDF vectorizer specifically for the recommender system and pre-computed a similarity matrix for all problems in the database. When you input a new problem, the system uses cosine similarity to find the 5 most similar problems and displays them in the sidebar. This helps users discover related practice problems.
>
> Second, I added **Automatic Tag Detection**. The system uses keyword matching to identify problem topics like 'Graph Theory', 'Dynamic Programming', 'Math', or 'Strings'. For example, if the description contains words like 'node', 'edge', or 'tree', it tags the problem as 'Graph'. This gives users quick insight into what algorithms they'll need."

---

## 1:50 - 2:10 : Challenges Faced

**Visual:** Show the confusion matrix or results section in README.

**Say:**
> "I faced two major challenges. First, **Score Calibration**. Initially, the web app artificially forced scores to match the predicted class—if it said 'Hard', the score was forced above 6.0. This hid the model's true confidence. I removed this artificial constraint so the classification and regression models operate independently, giving more honest predictions.
>
> Second, **Dataset Subjectivity**. The boundary between Medium and Hard is often blurred, which you can see in the confusion matrix where some overlap occurs. Despite this, the model achieves 51.24% accuracy, which is a meaningful improvement from the baseline 50.12%."

---

## 2:10 - 2:50 : Live Demo

**Visual:** Switch to the **Streamlit Web App** at localhost:8501.

**Action & Say:**

1. **Show Hard Example:**
   - Paste the graph cycle problem (Hard, Score ~7.77)
   - "Let's test with a Hard problem involving graph theory and special edge constraints."
   - Click **Predict Difficulty**
   - "As you can see, it correctly predicts 'Hard' with a score of 7.77. Notice it also detected the 'Graph' tag automatically. On the sidebar, you can see similar graph-related problems from the database."

2. **Show Medium Example:**
   - Clear and paste the mountain/camera problem (Medium, Score ~5.17)
   - "Now a Medium problem about geometry and visibility."
   - Click **Predict**
   - "It predicts 'Medium' with score 5.17, which is right in the Medium range. The tags show 'Geometry' and 'Math'."

3. **Highlight Sidebar:**
   - Point to the Similar Problems section
   - "Notice how the recommendations update based on the input. Each shows the similarity percentage and difficulty, helping users find related practice problems."

---

## 2:50 - End : Results & Conclusion

**Visual:** Show the **Results** section of README.md with metrics.

**Say:**
> "In conclusion, AutoJudge demonstrates that with proper noise reduction and feature engineering, we can automate difficulty assessment for competitive programming. The system achieves 51.24% classification accuracy and a Mean Absolute Error of just 1.68 for score prediction, meaning predictions are typically within 2 points of the true difficulty. The advanced features like similarity-based recommendations and automatic tagging make it a practical tool for problem setters and students alike. Thank you for watching."

---

## Key Points to Emphasize:
- ✅ **Noise Reduction**: LaTeX cleaning with regex
- ✅ **Trigrams**: 3-word phrases for better context
- ✅ **Similar Problems**: Cosine similarity with pre-computed matrix
- ✅ **Auto Tags**: Keyword-based topic detection
- ✅ **Honest Predictions**: Removed artificial score forcing
- ✅ **Metrics**: 51.24% accuracy, 1.68 MAE
