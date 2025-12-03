# Beyond Sampling: Explainable Credit Risk Modelling with SMOTE, ADASYN and SHAP

> **TL;DR**  
> This project is not “just another data sampling demo”.  
> It combines **multi-class credit risk modelling**, **SMOTE & ADASYN**, and **SHAP-based explainability + stability analysis** to show **how resampling changes not only metrics, but also how models reason about risk.**

---

## 1. Project Overview

Real-world credit risk data are almost always **imbalanced**:  
low-risk customers dominate the portfolio, while truly risky cases are rare.

Most tutorials stop at:

> “Apply SMOTE or ADASYN → check if accuracy/F1 improves → done.”

This project goes further.

We:

1. Build **three different classifiers** for a 3-level risk label (`Low`, `Moderate`, `High`):
   - **XGBoost**
   - **AdaBoost** (tree-based)
   - **ANN** (MLPClassifier)
2. Compare two oversampling techniques:
   - **SMOTE** (Synthetic Minority Over-sampling Technique)
   - **ADASYN** (Adaptive Synthetic Sampling)
3. Evaluate models with metrics that matter for imbalanced risk data:
   - ROC–AUC (macro + per-class)
   - Precision–Recall curves and **Average Precision (AP)**
   - F1, G-Mean, Sensitivity, Specificity
4. Use **SHAP** to answer a deeper question:

> When we change the sampling method,  
> do our models also **change the way they use features** to assign risk?

We then perform a **SHAP Stability Rate Analysis (SRA)** across different imbalance levels to see which sampling method yields **more stable and trustworthy explanations**.

---

## 2. Why this project is different

Most imbalanced-learning examples focus only on:

- “Which sampling method gives higher accuracy / F1?”

This repository explicitly studies **three dimensions at once**:

1. **Performance** – Does SMOTE or ADASYN improve metrics for each model?
2. **Interpretability** – How do SHAP feature importance and interactions change?
3. **Stability** – Are the explanations robust if the class imbalance changes?

In other words:

> We don’t only ask *“Which model scores higher?”*  
> We also ask *“Which model + sampling combination gives stable, interpretable reasons for its predictions?”*

This makes the project particularly relevant for:

- **Credit risk** and **regulatory settings** (where model transparency matters)  
- **Model governance / validation** teams  
- Anyone who wants to go **beyond black-box metrics** in imbalanced learning.

---

## 3. Dataset

- Input file: `FinalData_3_RiskLevels.xlsx`
- Target: `Risk_Level` (multi-class, 3 levels)

  | Code | Meaning        |
  |------|----------------|
  | 0    | Low risk       |
  | 1    | Moderate risk  |
  | 2    | High risk      |

- The classes are **highly imbalanced** (low risk dominates).
- Features include variables such as:
  - `TMDB_NUV`
  - `TMDB_Popularity`
  - `TMDB_Mean_UVR`
  - (plus additional behavioural / portfolio variables)

> **Note**  
> The raw dataset is not included for privacy reasons.  
> The notebooks assume the file is stored on **Google Drive** when running in Google Colab.

---

## 4. Repository Structure
```text
├── notebooks/
│   ├── 01_risk_datasampling_models.ipynb
│   │   # Baseline models + per-class metrics
│   └── 02_risk_shap_analysis.ipynb
│       # SMOTE vs ADASYN + SHAP + SRA
│
├── docs/
│   ├── SamplingTechniquesOnThreeModelsComparisonReport.docx
│   │   # Written report for supervisor
│   └── SHAPplots.docx
│       # Detailed SHAP interpretation notes
│
├── requirements.txt
└── README.md
```

## 5. Modelling and Evaluation

### 5.1 Baseline Models (Notebook: `01_risk_datasampling_models.ipynb`)

This notebook builds **three core models** without any oversampling:

1. **XGBoost**
   - Objective: multi-class classification (`multi:softmax`)
   - Hyperparameters tuned via `RandomizedSearchCV`, including:
     - `learning_rate`, `max_depth`, `gamma`
     - `colsample_bytree`
     - `reg_alpha`, `reg_lambda`

2. **AdaBoost**
   - Base estimator: `DecisionTreeClassifier`
   - Hyperparameters tuned with `RandomizedSearchCV`:
     - `n_estimators`, `learning_rate`
     - Tree depth and splitting criteria of the base estimator

3. **ANN (MLPClassifier)**
   - Hyperparameters tuned, e.g.:
     - `hidden_layer_sizes`
     - `activation`
     - `learning_rate_init`
     - `batch_size`
     - `max_iter`

#### Per-class metrics

For each model and each risk level (0 = Low, 1 = Moderate, 2 = High), the notebook computes:

- Accuracy  
- Sensitivity (Recall)  
- Specificity  
- F-measure  
- G-Mean  
- ROC–AUC  

All results are aggregated into a `results` DataFrame and used as the **baseline** for later comparison with sampled data.

---

### 5.2 Sampling & Model Comparison (Notebook: `02_risk_shap_analysis.ipynb`)

This notebook answers:

> “How do **SMOTE** and **ADASYN** change model performance and explanations?”

#### Sampling methods

- **SMOTE** – generates synthetic minority samples by interpolating between neighbours.
- **ADASYN** – similar to SMOTE but focuses more heavily on **hard-to-learn** minority samples.

For each method:

1. Apply oversampling **only on the training set**.
2. Train:
   - XGBoost
   - AdaBoost
   - ANN (MLPClassifier)
3. Evaluate on the **same untouched test set**.

#### Metrics

For each model + sampling combination, we compute:

- Overall Accuracy & Macro F1  
- ROC–AUC (per class & macro)  
- Precision–Recall curves  
- **Average Precision (AP)** per class  
- Confusion matrices (for qualitative inspection)

##### High-level findings

- **XGBoost**
  - Both SMOTE and ADASYN achieve **very high AUC** (~0.988–0.989).
  - ADASYN sometimes gives a **slightly higher Average Precision**, but the margin is small.
- **AdaBoost**
  - SMOTE and ADASYN perform similarly.
  - SMOTE tends to be **a bit more stable** across risk levels.
- **ANN**
  - For low- and high-risk classes, both methods perform very well.
  - For **moderate-risk (class 1)**, **SMOTE clearly outperforms ADASYN** in AUC and AP.
  - Overall, **SMOTE is the safer choice for ANN** on this dataset.

---

## 6. XAI with SHAP: Looking Inside the Models

Metrics alone can be misleading, especially in regulated domains like credit risk.  
We use **SHAP** to understand how sampling affects **feature importance and interactions**.

### 6.1 SHAP Summary Plots (XGBoost)

For the best XGBoost models under SMOTE and ADASYN, we:

- Use `shap.TreeExplainer` to compute SHAP values.
- Generate **summary plots** that show:
  - Global feature importance ranking.
  - Direction and magnitude of impact for each feature value.

**Observations**

- In both SMOTE and ADASYN, features like **`TMDB_NUV`** emerge as top drivers of risk.
- Under **SMOTE**, the ranking of secondary features (e.g. `TMDB_Mean_UVR`) is **more stable and consistent**.
- Under **ADASYN**, importance for features like `TMDB_Popularity` can increase, but SHAP distributions become more scattered – indicating a more complex, less smooth decision surface.

### 6.2 SHAP Interaction Plots

We inspect interactions such as **`TMDB_Popularity × TMDB_NUV`**:

- With **SMOTE**, interaction patterns are:
  - More structured.
  - Clusters of points form clear regions where risk increases.
- With **ADASYN**, interaction values are:
  - More spread out.
  - Suggest stronger local variability, potentially harder to explain to stakeholders.

This highlights that **different sampling methods do not only adjust class frequencies; they can also reshape how the model combines features to decide risk levels.**

---

## 7. SHAP Stability Rate Analysis (SRA)

To go one step further, we simulate **different imbalance levels** (e.g., changing default rates) and study **explanation stability**:

1. Resample the data to create multiple scenarios with varying class distributions.
2. For each scenario:
   - Fit XGBoost with SMOTE / ADASYN.
   - Compute SHAP values for the test set.
   - Extract top-k important features.
3. Compute a **Stability Rate** score:
   - How often do the same features remain in the top-k across scenarios?

### Key insight

- **SMOTE** tends to produce **higher SHAP stability scores**:
  - Top features remain consistent even as imbalance levels change.
- **ADASYN**:
  - Sometimes improves AP slightly, but its **feature rankings fluctuate more**.

From a **model governance / audit / regulator** perspective, **stable explanations** can be more important than tiny gains in AUC.

> **Takeaway:**  
> If the goal is **transparent and robust credit risk modelling**, SMOTE + XGBoost (or SMOTE + ANN) offers a more stable explanation pattern than ADASYN in this dataset.

---

## 8. Practical Recommendations

Based on the experiments:

- If you care purely about **tree-model performance**, both SMOTE and ADASYN are reasonable, with ADASYN providing small gains in some cases.
- If you need:
  - **Neural networks (ANN)**, **and/or**
  - **Stable, regulator-friendly explanations**,  
  **SMOTE is usually the better choice** on this dataset.
- SHAP should be used **alongside** metrics to validate that:
  - The model relies on **sensible features**, and
  - Those features remain **consistently important** across class distributions.



