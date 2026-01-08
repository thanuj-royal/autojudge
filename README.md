# AutoJudge ⚖️

AutoJudge is an intelligent system that predicts the difficulty class (Easy, Medium, Hard) and a numerical difficulty score (0-10) for programming problems based solely on their textual description.

## 📌 Approach

### Data Processing
- **Input**: Problem title, description, input/output description.
- **Preprocessing**: 
    - Combined all text fields into a single corpus.
    - **Noise Reduction**: Advanced cleaning to remove LaTeX commands (e.g., `\frac`, `\le`) while preserving meaningful text.
    - Standardized text (lowercasing, removing special characters).
- **Feature Extraction**: 
    - **TF-IDF**: Converted text into numerical vectors using `TfidfVectorizer`.
    - **Configuration**: Enabled **trigrams** (`ngram_range=(1,3)`) and increased `max_features` to 20,000 to capture complex phrases.

### Models
1.  **Classification (LinearSVC)**: Predicts difficulty class (`Easy`, `Medium`, `Hard`).
    - tuned with `class_weight='balanced'` to handle dataset imbalance.
2.  **Regression (RandomForestRegressor)**: Predicts a precise difficulty score (0-10).
    - Reuses the robust TF-IDF features for consistency.
    - Output is logically clipped to [0, 10].

## 📊 Results

We evaluated the models using 5-Fold Cross-Validation. The improved noise reduction and feature tuning yielded the following results:

### Classification Performance
**Accuracy**: **51.24%** (Improved from baseline 50.12%)

**Confusion Matrix**:
|        | Easy (Pred) | Medium (Pred) | Hard (Pred) |
|--------|-------------|---------------|-------------|
| **Easy**   | **68**      | 42            | 38          |
| **Medium** | 38          | **95**        | 141         |
| **Hard**   | 27          | 105           | **252**     |

*Observation: The model effectively distinguishes "Hard" problems but has some overlap between "Medium" and "Hard", which is expected given the subjective nature of difficulty.*

### Regression Performance
- **MAE (Mean Absolute Error)**: **1.68** (Significant improvement from 2.22)
- **RMSE (Root Mean Square Error)**: **1.98** (Improved from 2.73)

## 🎥 Demo Video
**Watch the demo:** [AutoJudge Demo Video](https://www.youtube.com/watch?v=Ya5b3HLl8Xo)

## 👤 Author Details
**Name**: Chakrala Reddy Thanuj Royal  
**ID**: 24114025  
**Branch**: CSE (2nd Year)

## 🛠️ Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Training (Optional)
To retrain the models from scratch:
```bash
python src/train_classifier.py
python src/train_regressor.py
```

### 3. Evaluation
To run the evaluation script and see current metrics:
```bash
python src/evaluate.py
```

### 4. Running the Web App
Start the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## 📁 Project Structure
- `app.py`: Main Streamlit application.
- `data/`: Contains the dataset.
- `models/`: Stores trained `.pkl` models.
- `src/`:
    - `train_classifier.py`: Trains the classification model.
    - `train_regressor.py`: Trains the regression model.
    - `evaluate.py`: Evaluates model performance.
    - `preprocess.py`: Shared data loading and cleaning logic.
