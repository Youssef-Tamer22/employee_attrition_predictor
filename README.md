# Employee Attrition Prediction

Predict which employees are likely to resign, using Logistic Regression and a Support Vector Classifier (SVC), and find out which factors (pay, performance, workload) are linked to leaving.

**Notebook:** `employee_attrition_simple.ipynb`
**Data:** `hr_employee_attrition.csv` (1,470 employees, 35 columns)

---

## 1. Quick start

1. Put `hr_employee_attrition.csv` in the same folder as the notebook.
   If the file has another name or location, change the path in cell 3:
   `data=pd.read_csv("hr_employee_attrition.csv")`
2. Install the requirements:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
   ```
   (Last run with Python 3, pandas 3.0.2, scikit-learn 1.8.0, seaborn 0.13.2. Other recent versions should work.)
3. Open the notebook and run all cells from top to bottom. The grid-search cells take the longest; allow a few minutes.

No other libraries are needed (no SMOTE / imbalanced-learn, no TensorFlow).

---

## 2. The problem

- **Target:** `Attrition` (Yes = the employee left). It is turned into `Attrition_bin` (1 = left, 0 = stayed).
- **Imbalance:** only **16.1%** of employees left (237 of 1,470), a 5.2 : 1 ratio. A model that always answers "stayed" would be about 84% accurate and catch nobody, so accuracy is **not** used to judge models.
- **Metrics used:** precision, recall and F1 for the *leaver* class, plus ROC-AUC and PR-AUC. Models are tuned on **PR-AUC** and the decision threshold is chosen to maximise leaver **F1**.

---

## 3. What the notebook does (30 cells, all code)

| Cells | Step |
|---|---|
| 1–2 | Imports, hide warnings |
| 3–6 | Load the data, `info()`, `describe()`, check missing values and duplicates (both are 0) |
| 7 | Drop `EmployeeCount`, `Over18`, `StandardHours` (constant) and `EmployeeNumber` (an ID); create the 0/1 target |
| 8 | **Plot:** class imbalance (bar chart and donut) |
| 9 | **Plot:** Spearman correlation heatmap for salary, performance and workload columns |
| 10 | **Plot:** distributions of the 8 numeric features most related to attrition |
| 11–12 | Count outliers (1.5 × IQR rule) and **plot** them before treatment |
| 13 | Split into features `X` and target `y` |
| 14–17 | Feature engineering and log transform (see below) |
| 18–19 | Import modelling tools; 80 / 20 stratified train–test split |
| 20–21 | Outlier capping class (`Winsorizer`) and the preprocessing pipeline |
| 22 | **Plot:** outliers after treatment |
| 23–24 | Grid search for Logistic Regression and for SVC |
| 25 | Choose the best decision threshold for each model |
| 26–27 | Test-set metrics table and classification reports |
| 28 | Model selection |
| 29–30 | **Plots:** confusion matrices; ROC and precision-recall curves |

### Feature engineering (cells 14–17)

- **Pay relative to the person:** income per job level, per year of experience, per year of age.
- **Career progress:** share of career spent at the company, years per previous company, time since last promotion, time in the same role, time with the current manager (all as ratios).
- **Satisfaction:** average of five satisfaction scores and a count of low scores.
- **Workload combinations:** overtime combined with being single, frequent travel or poor work-life balance, plus a combined `WorkloadStress` score.
- **Dropped:** `DailyRate`, `HourlyRate`, `MonthlyRate` (no relationship with attrition).
- **Ordinal scores** (job level, stock options, satisfaction, involvement, work-life balance) are kept as numbers **and** copied as categories, so the model can learn step changes.
- **Log transform:** `log1p` on continuous features with skew above 0.75.

### Outlier handling

Outliers are real employees (for example long-tenured or senior staff), so **no rows are deleted**. Instead:

1. Skewed features are log-transformed.
2. Remaining extreme values are capped at Q1 − 1.5·IQR and Q3 + 1.5·IQR. The limits are learned **only from the training data**, inside the pipeline, so nothing leaks from the test set.

Features such as `TrainingTimesLastYear` and `NumCompaniesWorked` have 10 or fewer distinct values, so they are treated as small whole-number scores and are scaled only (not capped or logged).

### Modelling choices

- **No data leakage:** scaling, capping and encoding are fitted inside cross-validation folds and on the training set only.
- **Grid search** (5-fold × 2 repeats, scored by PR-AUC):
  - Logistic Regression: penalty (l1 / l2), `C`, `class_weight`
  - SVC: linear or rbf kernel, `C`, `gamma`, `class_weight`
- **Threshold tuning:** for each model, out-of-fold training predictions are used to find the cut-off with the best leaver F1. The test set is not used for this.
- **Model selection:** best cross-validated PR-AUC wins; if the models are within one standard error of each other, the simpler one (Logistic Regression) is chosen. The test set is **not** used to choose the model.

---

## 4. Results (test set: 294 employees, 47 leavers)

| Metric | Logistic Regression | SVC |
|---|---|---|
| Accuracy | 0.854 | 0.867 |
| Precision (leavers) | 0.543 | 0.605 |
| Recall (leavers) | 0.532 | 0.489 |
| **F1 (leavers)** | **0.538** | **0.541** |
| ROC-AUC | 0.838 | 0.825 |
| PR-AUC | 0.600 | 0.591 |

- Best settings: Logistic Regression `C=1`, l1 penalty, no class weighting; SVC linear kernel, `C=0.1`.
- Cross-validated PR-AUC: 0.7073 (Logistic Regression) vs 0.7057 (SVC). The models are statistically tied, so Logistic Regression is selected as the simpler model.
- Both models rank leavers well above chance (the random baseline for PR-AUC is about 0.16).


## 6. Reproducibility

All random choices use `random_state=42` (train–test split, cross-validation, models), so re-running with the same data and library versions gives the same results.
