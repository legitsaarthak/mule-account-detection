# Mule Account Detection via Gradient Boosting

IIT Hyderabad × Bank of India Hackathon — PSB Cybershield, Problem Statement 2.

---

## Overview

A mule account is a bank account used to receive and forward stolen funds. The account holder is typically recruited unknowingly — often through fake job offers — and their account becomes a transit node in a money laundering chain. Detecting mule accounts is hard: fraudsters deliberately behave almost like legitimate customers to avoid rule-based detection.

This project builds a binary classifier that identifies mule accounts from a dataset of anonymised bank account features. The pipeline covers data cleaning, leakage removal, EDA, feature engineering, feature selection, baseline modelling, and a final XGBoost + SMOTE classifier with threshold tuning and SHAP-based interpretation.

---

## Dataset

- **File:** `iith boihackathon problem 2 data.csv` (not included — see below)
- **Size:** 9,082 accounts × 3,925 columns (F0–F3923 features + F3924 target)
- **Target:** F3924 — 1 = mule, 0 = legitimate
- **Class distribution:**
  - Legitimate: 9,001 accounts (99.11%)
  - Mule: 81 accounts (0.89%)
  - Imbalance ratio: approximately 111:1
- **Feature names:** Anonymous. Hackathon organisers provided 18 "hint features" (F115, F321, F527, ...) as a starting point. No semantic labels were given for any feature.

### Dataset placement

The dataset cannot be redistributed. To run the notebook, place the raw CSV in the same directory:

```
IITH BOIHackathon/
├── mule_account_detection.ipynb
├── iith boihackathon problem 2 data.csv   ← place here
├── requirements.txt
└── ...
```

---

## Approach

```
iith boihackathon problem 2 data.csv
  ↓
Drop columns with >50% missing values
Drop near-zero variance columns
Leakage Detection & Removal (F2230 — temporal feature encoding labelling month)
Encode categoricals (ordinal: F3889; label: F3886, F3890–F3893; date→age: F3888)
Impute remaining nulls with column medians
  ↓
EDA on 18 hint features
Ratio table (mule mean / legit mean) to identify signal despite right-skewed distributions
Correlation heatmap
  ↓
Feature Engineering
  - Ratio features: F670/F2678, F531/F321, etc.
  - Interaction products: F2678×F2122, F2678×F1692, F2122×F1692
  - Composite aggregate: sum_of_lows = F2678+F2122+F2956+F2737+F1692
  - Z-scores: F2678, F2122, F670, F2956, F2737
  ↓
Feature Selection
  - Random Forest (500 trees, class_weight='balanced') ranks all ~2000 features
  - Top 50 kept; sum_of_lows force-included (showed diagnostic signal despite missing top 50)
  ↓
Stratified train/test split (80/20, random_state=42)
  ↓
Baseline models
  - Logistic Regression (StandardScaler + class_weight='balanced')
  - Random Forest (class_weight='balanced', min_samples_leaf=3)
  ↓
XGBoost + SMOTE (ImbPipeline)
  - SMOTE: sampling_strategy=0.3, k_neighbors=5
  - XGBoost: n_estimators=500, learning_rate=0.05, max_depth=4,
             min_child_weight=3, subsample=0.8, colsample_bytree=0.8, gamma=1.0
  ↓
Threshold sweep (0.05 → 0.90) → optimal threshold = 0.45 (P=R crossover)
  ↓
Evaluation + Out-of-Fold CV (5-fold stratified)
  ↓
SHAP (TreeExplainer) — global importance + individual account waterfall plots
```

---

## Results

### Held-out test set (16 mules, 1,801 legitimate accounts)

| Metric | Value |
|--------|-------|
| Classification threshold | 0.45 |
| Mules caught | 13 / 16 |
| Recall | 81.2% |
| Precision | 81.25% |
| F1 | 0.8125 |
| False alarm rate | 0.22% (4 / 1,801 legitimate accounts flagged) |
| Investigation load | 1 confirmed mule per 1.3 analyst alerts |
| ROC-AUC | ~0.995 |

### Out-of-fold cross-validation (65 training mules — more reliable estimate)

| Metric | Value |
|--------|-------|
| OOF Recall | ~0.74 |
| OOF Precision | ~0.79 |
| OOF F1 | ~0.76 |
| OOF ROC-AUC | ~0.9649 |

The test-set numbers are slightly optimistic due to small sample size (16 mules). The OOF numbers are the more realistic expectation on unseen data.

### Baseline comparison (test set, threshold=0.5)

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.0505 | 0.9375 | 0.0958 | 0.9672 |
| Random Forest | 0.9000 | 0.5625 | 0.6923 | 0.9918 |
| **XGBoost + SMOTE (thr=0.45)** | **0.8125** | **0.8125** | **0.8125** | **~0.995** |

---

## Repository Structure

```
IITH BOIHackathon/
│
├── mule_account_detection.ipynb    ← Main notebook: full pipeline, analysis, results
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
│
├── src/  (original phase scripts — kept for reference)
│   ├── first_phase.py              ← Initial EDA: class distribution, hint features, object columns
│   ├── second_phase.py             ← Data cleaning, encoding, EDA plots, leakage finding
│   ├── third_phase_part_1.py       ← Feature engineering
│   ├── third_phase_part_2.py       ← RF feature selection, save X_model_ready.csv
│   ├── my_fourth_phase.py          ← Baseline models (LR + RF) + CV + ROC curves
│   ├── phase_five.py               ← XGBoost + SMOTE, threshold tuning, confusion matrix
│   ├── fifth_phase_and_shap.py     ← XGBoost + SMOTE + SHAP analysis
│
├── notes/  (development notes)
│   ├── phase2_notes.md             ← Reasoning behind cleaning and EDA decisions
│   ├── phase3_notes.md             ← Feature engineering theory and selection methodology
│   ├── phase4_notes.md             ← Baseline model reasoning and result interpretation
│   ├── phase5_notes.md             ← XGBoost parameter rationale, SMOTE reasoning
│   ├── phase5_new_notes.md         ← Extended phase 5 analysis including OOF comparison
│   └── phase6_shap_notes.md        ← SHAP findings: global importance and waterfall analysis
│
├── results/
│   ├── phase 4 results.txt         ← Baseline model terminal output
│   ├── xgb and smote results.txt   ← XGBoost comparison results
│   ├── Cross Validation results phase 5.txt  ← OOF CV output and confusion matrix
│   └── Threshold sweep of both XGB.txt      ← Full threshold sweep tables
│
├── plots/                          ← Generated figures
│   ├── hint_distributions.png
│   ├── hint_correlation.png
│   ├── roc_curve_baselines.png
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   ├── cm_xgboost+smote0.30.png
│   ├── precision_recall_threshold(XGB+SMOTE).png
│   ├── shap_bar.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_waterfall_missed.png
│   └── shap_waterfall_mule_number_*.png
│
└── mule_account_detection_whitepaper.pdf   ← Full project writeup
```


---

## Running the Project

### 1. Set up the environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Place the dataset

Copy `iith boihackathon problem 2 data.csv` into the project root directory (same folder as the notebook). The notebook uses a relative path and expects this exact filename.

### 3. Run the notebook

```bash
jupyter notebook mule_account_detection.ipynb
```

Run all cells from top to bottom. **Do not skip cells** — each section depends on variables set in earlier cells.

**Expected runtime:** ~10–20 minutes on a modern CPU (dominated by RF feature selection with 500 trees, XGBoost training with 500 estimators, and OOF cross-validation).

### Intermediate CSV files

The original scripts produced intermediate CSVs (`cleaned_data_with_target.csv`, `final_dataset_with_target.csv`, `X_model_ready.csv`, `y.csv`) that were used to pass data between phases. The notebook re-derives these in memory from the raw data — you do not need these files to run the notebook. If you want to regenerate them, run the original phase scripts in order.

---

## Limitations

**Small minority class:** The test set contains 16 mule accounts. Each caught or missed mule changes recall by 6.25 percentage points. Reported test-set metrics carry substantial uncertainty. The OOF estimate (65 training mules) is more reliable but still limited.

**Anonymous features:** All features are named F0–F3924. Interpretations (e.g., F3898 as "banking relationship depth") are based on value-distribution inspection and should not be treated as confirmed definitions.

**No network features:** The dataset contains no graph-level information — transfer recipients, multi-hop flows, holding times, or transaction network centrality. The missed mule analysis shows this is a real gap: mules disguising themselves through high transaction volume are likely to evade the current model.

**SMOTE from 65 real mules:** Training used ~2,100 synthetic mule accounts generated from 65 real ones. Synthetic examples may not represent novel mule strategies outside the training distribution.

**Hackathon dataset:** This data was prepared for a specific competition. Real production bank data would differ in scale, feature distribution, and mule behaviour. Results should not be generalised without validation on real-world data.

---

## Future Work

- **Graph/network features:** Transaction graph centrality, average holding time before forwarding, number of unique transfer recipients. The missed mule analysis points directly at this gap.
- **Temporal features:** Account-level velocity changes over time windows (30-day, 90-day), without leaking the labelling timestamp.
- **Proper threshold calibration:** Using a held-out validation set separate from the test set (difficult with 81 mules total, but correct practice).
- **More minority class data:** Most evaluation uncertainty stems from 81 total mules. Additional confirmed cases would enable reliable model comparison and more stable threshold selection.

---

## Whitepaper

`mule_account_detection_whitepaper.pdf` contains the full project writeup with methodology, results tables, SHAP plots, and limitations discussion.
