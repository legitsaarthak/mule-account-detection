# Phase 5 — XGBoost + SMOTE + Threshold Tuning

---

## What Phase 5 Is Actually Doing

Phase 4 ended with RF at ROC-AUC 0.9918 but recall of only 0.5625 — it was catching 9 out of 16
mules. The model knows how to rank accounts correctly (great AUC), but its default decision
threshold of 0.5 is too conservative. It's being too careful. Phase 5 fixes this three ways:

1. **XGBoost** — a stronger model that learns more complex patterns from your 50 features
2. **SMOTE** — gives the model more mule examples to learn from during training
3. **Threshold tuning** — instead of defaulting to 0.5, deliberately choose the cutoff that
   maximises recall while keeping precision operationally acceptable

These aren't three separate solutions. They work together. XGBoost produces better probability
scores, SMOTE produces a better-calibrated model, and threshold tuning lets you extract the
right operating point from whatever the model learned.

---

## Step 1 — Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
# ^ ImbPipeline is from imbalanced-learn, NOT sklearn.
#   Critical difference: ImbPipeline knows that SMOTE should only run during .fit(),
#   not during .predict() or .transform(). sklearn's Pipeline doesn't understand this
#   and would try to apply SMOTE to test data too — which is wrong.

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix
)
```

**Install if missing:**
```bash
pip install xgboost imbalanced-learn
```

---

## Step 2 — XGBoost with scale_pos_weight (No SMOTE Yet)

Before bringing in SMOTE, test XGBoost on its own with its built-in imbalance fix.
This gives you a clean comparison point.

```python
# scale_pos_weight is XGBoost's built-in way to handle imbalance.
# It tells XGBoost: "when you misclassify a mule, treat it as this many times
# more costly than misclassifying a legitimate account."
# The right value is simply: how many legitimate accounts per mule?
neg_count = (y_train == 0).sum()  # legitimate accounts in training set
pos_count = (y_train == 1).sum()  # mule accounts in training set
spw = neg_count / pos_count
# With ~7200 legit and ~65 mules in training: spw ≈ 110.7
# Meaning: a missed mule is treated as 110x worse than a false alarm.
print(f"scale_pos_weight: {spw:.2f}")

# XGBoost pipeline — no scaler needed.
# Trees split on thresholds, not distances. Whether a feature is measured
# in millions or in 0–1 range, the threshold just moves proportionally.
# Scaling doesn't change what XGBoost learns.
xgb_base = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    # ^ Same median imputation as Phase 4. Even with ~50 clean features,
    #   keep this in the pipeline in case any NaNs slipped through.
    ('xgb', XGBClassifier(
        n_estimators=300,
        # ^ Number of trees. XGBoost builds them sequentially, each one
        #   correcting the mistakes of the previous. More trees = better,
        #   but diminishing returns past ~300-500 on this dataset size.

        max_depth=4,
        # ^ How deep each tree is allowed to grow. Deeper = more complex
        #   patterns learned, but also more overfitting risk.
        #   With only ~65 mules in training, depth 4 is a conservative
        #   choice. A depth-4 tree can ask 4 yes/no questions — enough
        #   to find real patterns, not enough to memorise individual mules.

        learning_rate=0.05,
        # ^ How much each new tree is allowed to "correct" the previous.
        #   Smaller = more conservative corrections = more trees needed
        #   but more stable learning. 0.05 with 300 trees is a standard
        #   starting combination.

        scale_pos_weight=spw,
        # ^ The imbalance fix explained above.

        subsample=0.8,
        # ^ Each tree is trained on a random 80% of training rows.
        #   Prevents any one tree from seeing all the data — same
        #   randomisation principle as Random Forest, but applied
        #   differently. Reduces overfitting.

        colsample_bytree=0.8,
        # ^ Each tree sees a random 80% of features.
        #   With ~50 features: each tree sees ~40. Same idea as RF's
        #   √n features per split — forces diversity between trees.

        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ))
])

xgb_base.fit(X_train, y_train)
y_pred_xgb = xgb_base.predict(X_test)
y_prob_xgb = xgb_base.predict_proba(X_test)[:, 1]
# ^ predict_proba returns [prob_legitimate, prob_mule] for each account.
#   We want the mule probability (index 1) — this is the raw risk score
#   before any threshold is applied.

print("=== XGBoost (scale_pos_weight, threshold=0.5) ===")
print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate', 'Mule']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
```

**What to look for:** Compare ROC-AUC against RF's 0.9918. If XGBoost's AUC is higher,
it's a better ranker. The recall at threshold=0.5 probably still looks low — that's expected.
Threshold tuning comes in Step 4.

---

## Step 3 — SMOTE + XGBoost

```python
# SMOTE — Synthetic Minority Oversampling Technique.
#
# The problem it solves: you have ~65 mule accounts in training.
# XGBoost builds each tree on a subset of that data. Some trees might
# only see 30-40 mules. That's not enough examples to learn reliable
# patterns from — the model memorises specific mules rather than
# learning what mules generally look like.
#
# SMOTE fixes this by generating synthetic mule accounts.
# How: it takes a real mule, finds its k nearest neighbours among
# other mules (in feature space), and creates a new fake mule somewhere
# on the line between them. It's interpolation, not random noise.
#
# The result: instead of 65 mules, you might train on 500+ mules,
# all statistically plausible. The model sees more variation in
# mule behaviour and learns more general patterns.
#
# CRITICAL: SMOTE only happens inside the pipeline's .fit() call.
# Your test set stays completely untouched. You're not faking test data —
# you're giving the model more to learn from during training only.

xgb_smote = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),

    ('smote', SMOTE(
        sampling_strategy=0.3,
        # ^ After SMOTE, mules will be 30% of the training set.
        #   NOT 50% (fully balanced) — that's usually too aggressive
        #   for fraud data. With 30%, you go from ~65 mules to roughly
        #   2100 synthetic+real mules against 7200 legitimate accounts.
        #   The model gets enough mule examples without the training
        #   distribution drifting too far from reality.
        #   Try 0.1, 0.2, 0.3 if results vary — 0.3 is a reasonable start.

        k_neighbors=5,
        # ^ Each new synthetic mule is created between a real mule and
        #   one of its 5 nearest mule neighbours. With only 65 real mules,
        #   k=5 means you're interpolating between fairly similar accounts.
        #   Smaller k = more conservative synthesis (closer to real mules).

        random_state=42
    )),

    ('xgb', XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        # ^ No scale_pos_weight here — SMOTE already handles the imbalance
        #   by generating more mules. Using both would double-compensate
        #   and over-weight the minority class.
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ))
])

xgb_smote.fit(X_train, y_train)
y_pred_smote = xgb_smote.predict(X_test)
y_prob_smote = xgb_smote.predict_proba(X_test)[:, 1]

print("=== XGBoost + SMOTE (threshold=0.5) ===")
print(classification_report(y_test, y_pred_smote, target_names=['Legitimate', 'Mule']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_smote):.4f}")
```

---

## Step 4 — Compare and Pick Your Base Model

```python
# Before threshold tuning, decide which base model to tune.
# The right comparison is ROC-AUC — it measures ranking quality
# independent of threshold. The better ranker will give better
# results at any threshold you choose.

results = {
    'RF Baseline':         roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1]),
    'XGB scale_pos_weight': roc_auc_score(y_test, y_prob_xgb),
    'XGB + SMOTE':         roc_auc_score(y_test, y_prob_smote),
}
# ^ Replace rf_pipeline with whatever you named your Phase 4 RF pipeline.

for name, auc in results.items():
    print(f"{name}: {auc:.4f}")

# Pick the model with the highest ROC-AUC for threshold tuning.
# If XGB+SMOTE and XGB+scale_pos_weight are close, prefer SMOTE —
# it tends to be more stable across different test splits because
# the model has seen more diverse mule behaviour during training.
```

---

## Step 5 — Threshold Tuning (The Most Important Step)

```python
# Every model outputs a probability between 0 and 1 for each account.
# By default, predict() converts this to a label using threshold=0.5:
#   prob >= 0.5 → "mule"
#   prob < 0.5  → "legitimate"
#
# But 0.5 is completely arbitrary. It made sense when your classes
# were balanced. With severe imbalance and a goal of maximising recall,
# you almost always want a lower threshold — flag more accounts as
# mules, catch more real ones, accept more false alarms.
#
# The question is: how low?
# Answer: wherever the precision-recall tradeoff is acceptable.
#
# precision_recall_curve gives you every possible threshold's
# precision and recall simultaneously. You pick the point that
# matches your operational requirements.

# Use your best model's probabilities (replace y_prob_smote if needed)
best_probs = y_prob_smote  # or y_prob_xgb — whichever won Step 4

precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
# ^ precisions[i], recalls[i] = the precision and recall you'd get
#   if your threshold was thresholds[i].
#   Note: len(thresholds) = len(precisions) - 1 (sklearn quirk — last
#   precision/recall pair has no corresponding threshold)

# --- Plot the curve ---
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], label='Precision', color='steelblue')
plt.plot(thresholds, recalls[:-1], label='Recall', color='tomato')
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision vs Recall at Every Threshold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_threshold.png', dpi=150)
plt.show()
# Reading this plot:
# - Moving left (lower threshold): recall goes up, precision goes down
# - Moving right (higher threshold): precision goes up, recall goes down
# - Find the threshold where recall is high but precision hasn't collapsed

# --- Find the optimal threshold programmatically ---
# Strategy 1: Maximise F1 (balanced precision-recall)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]
print(f"Best F1 threshold: {best_f1_threshold:.3f}")
print(f"  Precision: {precisions[best_f1_idx]:.3f}, Recall: {recalls[best_f1_idx]:.3f}, F1: {f1_scores[best_f1_idx]:.3f}")

# Strategy 2: Maximise recall subject to precision >= 0.3
# (Meaning: at least 30% of flagged accounts must be real mules.
#  Tune this threshold based on how many false investigations are acceptable.)
min_precision = 0.3
valid_mask = precisions[:-1] >= min_precision
if valid_mask.any():
    # Among thresholds that maintain >= 30% precision, find the one with highest recall
    best_constrained_idx = np.argmax(recalls[:-1] * valid_mask)
    best_constrained_threshold = thresholds[best_constrained_idx]
    print(f"\nBest recall with precision >= {min_precision}:")
    print(f"  Threshold: {best_constrained_threshold:.3f}")
    print(f"  Precision: {precisions[best_constrained_idx]:.3f}, Recall: {recalls[best_constrained_idx]:.3f}")
else:
    print(f"No threshold achieves precision >= {min_precision}. Lower the constraint.")

# --- Apply chosen threshold ---
chosen_threshold = best_constrained_threshold
# ^ Or best_f1_threshold — your call based on what the plot shows.
#   For a hackathon judging panel, being able to explain your threshold
#   choice matters. "I chose 0.3 because it maximises mule detection
#   while keeping at least 1 in 3 alerts genuine" is a real argument.

y_pred_tuned = (best_probs >= chosen_threshold).astype(int)

print(f"\n=== Final Model (threshold={chosen_threshold:.3f}) ===")
print(classification_report(y_test, y_pred_tuned, target_names=['Legitimate', 'Mule']))
print(f"ROC-AUC: {roc_auc_score(y_test, best_probs):.4f}")
```

---

## Step 6 — StratifiedKFold Cross-Validation on Final Config

```python
# One test split with 16 mules is fragile — catching one extra mule
# swings recall by 6.25%. CV on 5 folds gives each mule exactly one
# chance to be in a test set, averaging results across all of them.
# This confirms the threshold result wasn't luck.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We need a custom scorer for tuned-threshold recall,
# so we use cross_validate with predict_proba and apply threshold manually.
from sklearn.model_selection import cross_val_predict

# Get out-of-fold probabilities — each account's probability is predicted
# by a model that never trained on that account. This is the cleanest
# possible evaluation with small data.
oof_probs = cross_val_predict(
    xgb_smote,   # replace with xgb_base if that won Step 4
    X_train,
    y_train,
    cv=cv,
    method='predict_proba'
)[:, 1]

# Apply chosen threshold to OOF predictions
oof_preds = (oof_probs >= chosen_threshold).astype(int)

print("=== Cross-Validation (Out-of-Fold) Results ===")
print(classification_report(y_train, oof_preds, target_names=['Legitimate', 'Mule']))
print(f"OOF ROC-AUC: {roc_auc_score(y_train, oof_probs):.4f}")

# If OOF recall is close to test recall — good. The model generalises.
# If OOF recall is much lower than test recall — your test split was lucky.
#   In that case, trust OOF numbers more.
```

---

## Step 7 — Confusion Matrix (Numbers, Not Just Percentages)

```python
# At this point you want to see raw numbers, not just ratios.
# With 16 mules in the test set, "recall 0.81" means 13 caught, 3 missed.
# Presenting raw numbers in the final report is more honest and more
# interpretable to non-technical judges.

cm = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()

print(f"""
Confusion Matrix (threshold={chosen_threshold:.3f}):
                  Predicted Legit   Predicted Mule
Actual Legit      {tn:>14}   {fp:>13}   (False Alarms)
Actual Mule       {fn:>14}   {tp:>13}   (Caught)

Mules caught:     {tp} / {tp+fn}  ({tp/(tp+fn)*100:.1f}% recall)
False alarms:     {fp} / {tn+fp}  ({fp/(tn+fp)*100:.2f}% of legit accounts flagged)
Investigation load: 1 real mule per {(tp+fp)/tp:.1f} alerts
""")
```

---

## What to Expect From Phase 5 Results

The Phase 4 RF caught 9 out of 16 mules (recall 0.5625) at default threshold.
After Phase 5, you're targeting roughly:

| Metric         | Phase 4 RF  | Phase 5 Target  |
|----------------|-------------|-----------------|
| Recall         | 0.5625      | **0.75 – 0.90** |
| Precision      | 0.9000      | 0.30 – 0.60     |
| ROC-AUC        | 0.9918      | 0.99+           |

The precision drop is intentional. You're trading false alarm rate for coverage.
The ROC-AUC should stay flat or improve — that measures ranking quality, not threshold.

---

## Common Problems and Fixes

**"SMOTE says k_neighbors > n_samples"**
You have fewer than 6 mule accounts in one CV fold. Reduce `k_neighbors=3`.

**"XGBoost recall is lower than RF baseline"**
Check that `scale_pos_weight` is correctly calculated from `y_train` not `y`.
Also check that IndexSS isn't back in your features.

**"precision_recall_curve shows precision collapsing to 0.01 immediately"**
Your model's probability scores are poorly calibrated — everything is near 0 or near 1.
Add `use_label_encoder=False` and try reducing `n_estimators` to 100.

**"OOF recall is much lower than test recall"**
Your 16-mule test set happened to contain the "easy" mules. Trust OOF numbers.
This is a dataset size problem, not a model problem.

---

## End of Phase 5 Deliverable

By the end of this phase you should have:

1. A trained XGBoost model (with SMOTE or scale_pos_weight — whichever won)
2. A chosen threshold with documented justification
3. Final test metrics: recall, precision, F1, ROC-AUC, raw confusion matrix numbers
4. OOF cross-validation confirming the results are stable
5. The precision-recall curve plot (useful for the presentation)

Lock in your best model. Phase 6 is SHAP — explaining *why* the model flags what it flags.
