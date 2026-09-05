# Phase 6 — SHAP Findings

---

## What SHAP Actually Does

The model outputs a probability for each account. SHAP explains *why* — for any individual
account, it breaks the prediction down into each feature's contribution. Positive SHAP value
= pushed toward mule. Negative SHAP value = pushed toward legitimate. All contributions sum
to: final prediction minus the baseline (average prediction across all accounts).

TreeExplainer is used specifically for tree-based models like XGBoost. It's exact, not an
approximation.

---

## Global Findings — Bar Plot and Beeswarm

**F3898 dominates everything.** Its mean absolute SHAP value (~2.0) is nearly 3x the next
most important feature (F3908 at ~0.75). No other feature comes close.

Top 5 features by importance:
1. F3898 — banking relationship depth (~2.0)
2. F3908 (~0.75)
3. F270 (~0.52)
4. F1921 (~0.50)
5. F2489 (~0.50)

`sum_of_lows` (the engineered composite feature from Phase 3) appears at position 12 —
it contributed meaningfully. The feature engineering was not wasted.

The original hint features (F2678, F2122, F2956 etc.) do not appear in the top 20 by name.
They were either absorbed into `sum_of_lows` or ranked below position 20. The model found
stronger signal in features discovered independently during Phase 3.

---

## Reading the Beeswarm

**F3898:** Red dots (high values) cluster far left at -3 to -4. High F3898 strongly pushes
accounts toward legitimate. Low F3898 (blue) sits near zero or slightly positive. Translation:
legitimate accounts have deep banking relationships. Mule accounts don't. This is the model's
primary decision rule.

**F3908:** Blue dots spread right. Low F3908 pushes toward mule. High F3908 (red) near zero.
This feature flags mules by absence — low values are the signal, not high ones.

**sum_of_lows:** Dots cluster on the left side — low sum_of_lows pushes toward mule, which
is consistent with Phase 2's finding that mule accounts had systematically lower values across
F2678, F2122, F2956, F2737, and F1692.

---

## Individual Account Analysis — Waterfall Plots

### Missed Mule — Account #98

- Final prediction f(x) = -2.257 (strongly toward legitimate)
- Baseline E[f(x)] = -1.323
- Model moved *further* from mule than average — not toward it

**Why it was missed:**
F950 = 399,410 pushed -1.03 toward legitimate. F1921 = 34,873 pushed -0.89. This mule had
transaction volumes indistinguishable from a high-activity legitimate account. Even though
F3898=1 (+0.68), F3908=1 (+0.71), F2489=0.06 (+0.72) and F270=0.95 (+0.66) all nudged
toward mule, 42 other features combined for -0.96, and the large transaction volume features
buried the mule signal entirely.

This mule disguised itself through high legitimate-looking activity. The model had no
distinguishing information available to catch it.

---

### Caught Mule — Account #1460 (most significant)

- Final prediction f(x) = -0.166 (log-odds)
- Converting to probability: sigmoid(-0.166) ≈ **0.459**
- Threshold = 0.45
- **Caught by a margin of 0.009**

At the default threshold of 0.50, this account is missed. Threshold tuning from 0.50 to
0.45 is the sole reason this mule is in the caught column.

**The internal battle:**
F3908 = 0 alone pulled -1.96 toward legitimate — a single feature almost single-handedly
hiding the account. Against it: F2489=0 (+0.61), F270=1 (+0.60), F3811=126847 (+0.47),
F1165=24966 (+0.46), F1819=126847 (+0.39), F1166=22000 (+0.36), F1921=1838 (+0.30), and
42 other features combined (+0.73). The coalition of smaller features collectively overcame
the dominant negative pull and pushed the final score just past 0.45.

**Why this matters for the presentation:**
This single account demonstrates three things simultaneously:
1. The model catches subtle mules that don't look obviously fraudulent
2. Threshold tuning had a real, measurable impact — 1 out of 13 caught mules exists
   solely because of deliberate optimisation from 0.50 to 0.45
3. The explainability is granular enough to show exactly which features saved the catch
   and which nearly caused a miss

---

## One-Line Summary for Judges

*"The dominant predictor is banking relationship depth — mule accounts show systematically
shallow banking histories, which the model weights almost 3x more heavily than any other
feature. At the individual account level, SHAP allows us to explain every flag and every
miss in terms of specific measurable behaviours."*
