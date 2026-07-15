# Titanic Survival Prediction

A binary classification project predicting whether a Titanic passenger survived, using the classic Kaggle *Titanic — Machine Learning from Disaster* dataset.

## Dataset

| Item | Detail |
|------|--------|
| Source | [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) |
| Training data | 891 rows, 12 columns |
| Test data | Separate (`test.csv`), no `Survived` column |
| Target | `Survived` (0 = did not survive, 1 = survived) |
| Evaluation metric | Accuracy |

## Project Structure

```
├── train.csv               # Training data
├── test.csv                # Test data (for submission)
├── titanic.ipynb            # Main notebook: EDA + preprocessing + modeling
├── titanic.csv               # Prediction output (submission)
└── README.md
```

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked data structure, missing values, and duplicates.
- Target distribution: mild class imbalance (61.6% did not survive vs 38.4% survived).
- Outlier detection (IQR method) on `Age` and `Fare` — outliers were kept since they were still domain-plausible.
- Numerical correlation (`Pclass`, `Fare`, etc.) and Chi-Square test for categorical features against `Survived`.

**Key findings:**
| # | Finding |
|---|---------|
| 1 | `Sex` is the most significant categorical predictor (Chi-Square p ≈ 0.0000) |
| 2 | `Pclass` has a fairly strong negative correlation with `Survived` (r ≈ -0.34) and correlates strongly with `Fare` (r ≈ -0.55) |
| 3 | `Cabin` is not significant against `Survived` (p ≈ 0.18) and has 77.1% missing values → dropped |
| 4 | `Fare` is right-skewed → handled with binning (`qcut`, 5 categories) |
| 5 | `Age`, `SibSp`, `Parch` show weak individual correlation → motivated the creation of combined features |

### 2. Preprocessing & Feature Engineering
- **Title** extracted from the `Name` column, used to impute `Age` based on the median per title.
- **Family_Size** = `SibSp` + `Parch`.
- **Family_Survival** built by grouping passengers with the same `Last_Name` + `Fare`, and the same `Ticket`, capturing the pattern that "one family tends to survive together" (computed without data leakage — only using information from other family members).
- **FareBin_Code** and **AgeBin_Code**: result of binning (`qcut`) + label encoding.
- Final features used by the model: `Pclass`, `Sex`, `Family_Size`, `Family_Survival`, `FareBin_Code`, `AgeBin_Code`.
- `StandardScaler` is fit only on the training data, then applied (*transform*) to the test data (no leakage).

### 3. Modeling
- Algorithm: **K-Nearest Neighbors (KNN)**.
- Hyperparameter tuning with `GridSearchCV` (10-fold cross-validation, `roc_auc` scoring).
- Grid: `n_neighbors [6–22]`, `weights [uniform, distance]`, `leaf_size [1–46]`, `algorithm [auto]`.

**Best result:**
| Aspect | Result |
|--------|--------|
| Best CV Score | ROC-AUC 0.8818 |
| Best params | `n_neighbors=14`, `leaf_size=26` |

### 4. Submission
The final model predicts on `test.csv`, and results are saved to `titanic.csv` in `PassengerId, Survived` format as required by Kaggle.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
```

## How to Run

1. Place `train.csv` and `test.csv` in the same directory as the notebook.
2. Run `titanic.ipynb` from top to bottom.
3. The submission file will automatically be saved as `titanic.csv`.

## Notes & Next Steps

- The current CV scoring uses **ROC-AUC**, while the Kaggle leaderboard is scored on **accuracy** — consider switching to `scoring='accuracy'` in `GridSearchCV` so the tuning aligns with the competition metric.
- Compare KNN's performance against other models (Logistic Regression, Random Forest, Gradient Boosting) to see if there's room for improvement.
- Add explicit `StratifiedKFold` to ensure consistent class proportions across folds.
- Further evaluate using a confusion matrix to inspect the precision/recall trade-off, not just accuracy/ROC-AUC.
