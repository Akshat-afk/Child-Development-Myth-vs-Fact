# Child Development Fact vs. Myth Classifier

## 📌 Project Overview
This project implements a **Child Development Fact vs. Myth classification system** using **Bag-of-Words (BoW) with n-grams (1 to 3)** and a **Multinomial Naive Bayes (MNB) model**.  
The goal is to automatically classify statements as **"Fact"** or **"Myth"** based on their textual content.

Key features:
- Text preprocessing with stopword removal and contraction handling.
- Feature extraction using **BoW (binary, uni-grams to tri-grams)**.
- Feature selection via **Mutual Information**.
- Model optimization with **GridSearchCV** for smoothing (`alpha`).
- Evaluation using **accuracy, classification report, confusion matrix, and 10-fold cross-validation**.

---

## ⚙️ Workflow
### 1. Data Preprocessing
- Expands contractions (e.g., "can't" → "cannot").
- Removes non-alphabetic characters.
- Converts text to lowercase.
- Removes **English stopwords**.

### 2. Feature Extraction
- **BoW model** with:
  - n-grams: (1, 3)
  - binary=True (presence/absence of words)

### 3. Feature Selection
- **SelectKBest + Mutual Information (k=9000)**

### 4. Model Training
- **Multinomial Naive Bayes** with smoothing parameter (`alpha`).
- **GridSearchCV** (5-fold) used to find the best `alpha`.

### 5. Evaluation
- **Accuracy, Precision, Recall, F1-score**
- **Confusion Matrix**
- **10-fold Cross-validation with 95% Confidence Interval**
- Inference speed: ~2.16 µs per statement.

---

## 📊 Results
- **Best alpha (ε):** `0.1`  
- **Accuracy:** `96.06%`  
- **Classification Report:**
  - *Myth*: Precision=1.00, Recall=0.88, F1=0.93
  - *Fact*: Precision=0.95, Recall=1.00, F1=0.97
- **Confusion Matrix:**
  ```
  [[ 77  11]
   [  0 191]]
  ```
- **10-fold CV Accuracy:** `97.41%`  
- **95% CI:** `(95.38%, 99.45%)`

---

## 🚀 How to Run
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure you have **NLTK stopwords and wordnet**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
3. Run the script:
   ```bash
   python classifier.py
   ```

---

## 📦 Dependencies
- `nltk`
- `scikit-learn`
- `pandas`
- `contractions`
- `numpy`

---

## 📌 Notes
- Dataset is expected in a **Pandas DataFrame** with:
  - `Statements` → input text
  - `Label` → target class (`Fact` or `Myth`)
- Feature selection parameter `k=9000` may be tuned based on dataset size.
- The model is lightweight and achieves **high accuracy with fast inference**, making it suitable for real-time applications.

---
