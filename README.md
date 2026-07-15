# Titanic Survival Prediction

This repository contains a machine learning project focused on predicting passenger survival aboard the Titanic using the classic Kaggle *Titanic — Machine Learning from Disaster* dataset. The dataset includes demographic, ticket, and cabin information for 891 passengers. Core attributes include passenger class, sex, age, fare, and family relations. The main goal is to build a classification model that accurately predicts whether a passenger survived.

---

## 📌 Problem Statement

Predicting survival outcomes from historical disaster data is a classic benchmark for classification workflows involving missing data, feature engineering, and imbalanced classes. This project aims to:

- Develop a classification model to predict `Survived` using demographic and ticket-related features.
- Handle missing data (especially `Age` and `Cabin`) using informed imputation rather than naive dropping.
- Engineer features that capture family-level survival patterns (`Family_Size`, `Family_Survival`) beyond individual-level attributes.
- Evaluate and tune the model using cross-validation to select the best-performing hyperparameters.

---

## 🧠 Features

The dataset contains the following features:

| Feature Name | Description | Type |
|---|---|---|
| `Pclass` | Passenger ticket class (1st, 2nd, 3rd) | Categorical |
| `Sex` | Passenger gender | Categorical |
| `Age` | Passenger age in years (imputed via median per Title) | Numerical |
| `SibSp` | Number of siblings/spouses aboard | Numerical |
| `Parch` | Number of parents/children aboard | Numerical |
| `Fare` | Ticket fare paid | Numerical |
| `Family_Size` | Engineered: `SibSp` + `Parch` | Numerical |
| `Family_Survival` | Engineered: survival signal from family/ticket group | Numerical |
| `Survived` | Target variable: whether the passenger survived | Binary |

---

## 🧪 Project Structure

```
titanic/
├── data/
│   ├── train.csv             # Training data with Survived labels
│   └── test.csv               # Test data for submission
├── models/
│   └── model_final.pkl        # Saved trained model (optional)
├── notebooks/
│   └── titanic.ipynb            # EDA, preprocessing, feature engineering, and modeling
├── output/
│   └── titanic.csv               # Final submission predictions
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md                 # Project documentation
```

---

## 🔁 Workflow

This project follows a typical machine learning workflow:

1. **Data Collection and Preparation**
   - Downloaded from Kaggle (see [Dataset & Credits](#-dataset--credits) section).
   - Train and test sets are provided separately by the competition.

2. **Data Preprocessing**
   - Handled missing values in `Age` (median per extracted `Title`) and dropped `Cabin` (77% missing, not statistically significant against `Survived`).
   - Checked for duplicates (none found) and outliers via IQR (kept, as values were domain-plausible).

3. **Exploratory Data Analysis (EDA)**
   - Analyzed target distribution, feature distributions, and outliers.
   - Visualized numerical correlation and ran Chi-Square tests for categorical features against `Survived`.

4. **Feature Engineering**
   - Extracted `Title` from `Name` for age imputation.
   - Built `Family_Size` and `Family_Survival` to capture family-level survival patterns.
   - Binned `Fare` and `Age` into categories, then label-encoded them.

5. **Model Training**
   - Trained a **K-Nearest Neighbors (KNN)** classifier on the engineered feature set.
   - Hyperparameter tuning performed using `GridSearchCV` (10-fold cross-validation, `roc_auc` scoring).

6. **Model Evaluation**
   - Evaluated using cross-validated ROC-AUC.
   - Best-performing configuration: **KNN (n_neighbors=14, leaf_size=26)**.

---

## 📈 Model Performance

A single model family (KNN) was tuned across a hyperparameter grid. The best configuration was selected based on cross-validated ROC-AUC:

| Model | CV ROC-AUC | Best Parameters |
|---|---|---|
| KNN (tuned) | **0.8818** | `n_neighbors=14`, `leaf_size=26`, `weights` and `algorithm` selected via grid search |

**Note:** cross-validation scoring used ROC-AUC, while the Kaggle leaderboard is scored on accuracy — the two metrics don't always align, so the leaderboard score may differ from the CV result above.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from the Kaggle Titanic competition. You can access the original dataset and challenge description via the following link:

🔗 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

We would like to extend our appreciation to Kaggle for making this resource available for public use.

---

## 🚀 How to Run

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```
git clone https://github.com/<your-username>/titanic.git
cd titanic
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

Make sure you have Jupyter installed, then run:

```
jupyter notebook notebooks/titanic.ipynb
```
