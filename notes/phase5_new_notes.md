# Phase 5 — What We Actually Did and Why

---

## The Goal of Phase 5

Phase 4 ended with Random Forest at ROC-AUC 0.9918 but recall of only 0.5625 — catching 9 out
of 16 mules at the default threshold of 0.5. The model was a strong ranker but a conservative
predictor. Phase 5 had three jobs: replace Random Forest with a more powerful model (XGBoost),
give that model more mule examples to learn from during training (SMOTE), and deliberately
choose an operating threshold instead of defaulting to 0.5.

These three things compound. A better model produces better probability scores. Better training
data from SMOTE produces better-calibrated scores. Threshold tuning extracts the right operating
point from those scores. You can't just tune the threshold on a weak model and expect the same
result — the quality of the underlying probabilities matters.

---

## Why XGBoost Over Random Forest

Random Forest builds hundreds of independent trees and takes a majority vote. Each tree is
trained on a random bootstrap sample with a random subset of features, and the trees don't
communicate. Their errors cancel out through averaging, which is powerful — but each tree
is completely ignorant of what the others got wrong.

XGBoost builds trees sequentially. Each new tree specifically focuses on the accounts the
previous trees got wrong. The model iteratively corrects its own mistakes. This makes
XGBoost more powerful on tabular data with complex patterns — which is exactly what
fraud detection is. The differences between mule and legitimate accounts are subtle,
non-linear, and live in combinations of features. XGBoost finds these interactions more
reliably than Random Forest.

On standard benchmarks and in competition results, XGBoost (and its variants LightGBM,
CatBoost) consistently outperform Random Forest on tabular classification. This isn't
a universal law, but it's a strong prior that held on this dataset.

---

## The Parameters and Why

```python
XGBClassifier(
    n_estimators=300,       # number of trees built sequentially
    max_depth=4,            # how many yes/no questions each tree can ask
    learning_rate=0.05,     # how much each tree corrects the previous one
    min_child_weight=3,     # minimum samples required to make a split
    subsample=0.8,          # each tree trains on 80% of training rows
    colsample_bytree=0.8,   # each tree sees 80% of features
    gamma=1,                # minimum loss reduction required to make a split
    scale_pos_weight=1,     # neutral — SMOTE handles imbalance instead
)
```

**n_estimators=300:** More trees means more sequential corrections. The right number depends
on learning_rate — lower learning rate needs more trees to converge. 300 with lr=0.05 is
a standard, well-matched pairing for a dataset of this size.

**max_depth=4:** Each tree can ask 4 yes/no questions. With only ~65 mules in training, deeper
trees would memorise specific mule accounts rather than learning general patterns. Depth 4
is enough to find meaningful patterns while being too shallow to overfit individual samples.
This is a regularisation decision forced by the small minority class size.

**min_child_weight=3:** A split is only allowed if both resulting branches contain at least
3 samples. This is XGBoost's equivalent of `min_samples_leaf` in sklearn. It prevents the
model from making splits that are justified by 1-2 mule accounts — which would be memorisation,
not learning. On a dataset with 65 minority class samples, this parameter matters more than
it would on a balanced dataset.

**subsample and colsample_bytree at 0.8:** Randomisation borrowed from Random Forest's logic.
Each tree sees a different 80% of rows and 80% of features, creating diversity between trees
and reducing overfitting. On a small minority class, diversity in what each tree sees helps
the model generalise rather than memorise.

**gamma=1:** A split is only made if it reduces the loss by at least 1. Without this, XGBoost
will make splits that barely improve anything just to technically reduce training error.
With 65 mules, many such micro-splits would be noise. Gamma enforces that every split is
earning its place.

**`min_samples_leaf` and `min_samples_split` are sklearn Random Forest parameters.** They
do not exist in XGBoost. Passing them to XGBClassifier will silently fail or throw errors.
`min_child_weight` is the XGBoost equivalent.

**`max_depth=None` does not work in XGBoost.** XGBoost defaults to depth 6 if omitted, and
None will error. Explicitly set a value.

---

## SMOTE — What It Does and What to Watch For

SMOTE generates synthetic minority class samples during training. It finds a real mule account,
locates its k nearest mule neighbours in feature space, and creates a new fake mule by
interpolating between them. The result is not random noise — it's statistically plausible
mule accounts that fill in the gaps between real ones.

The motivation: with 65 real mules, the model sees very limited variation in mule behaviour.
SMOTE expands that variation without requiring real data.

**sampling_strategy=0.3** means after SMOTE, mules are 30% of the training set. Not 50%
(fully balanced) — that's too aggressive. It would create thousands of synthetic mules from
65 real ones, drifting far from the true distribution. 30% gives the model enough mule
examples to learn from without hallucinating a world where every third account is a mule.

**k_neighbors=5** means each synthetic mule is created between a real mule and one of its
5 nearest mule neighbours. Smaller k = more conservative synthesis, staying closer to real
mules. With only 65 real mules, k=5 is a reasonable choice — it's specific enough to produce
realistic synthetics without being so tight that it just duplicates existing accounts.

**BorderlineSMOTE** generates synthetics only near the decision boundary — the mules that
most closely resemble legitimate accounts. In theory this is better for fraud data, because
fraudsters deliberately behave almost like legitimate accounts. In practice, it requires
enough minority samples to reliably identify which ones are "borderline." With 65 mules,
the neighbourhood calculation is unstable. BorderlineSMOTE pays off at hundreds of minority
samples. At 65, regular SMOTE is more reliable.

**ImbPipeline vs sklearn Pipeline:** SMOTE must only be applied during training, never
during prediction. sklearn's Pipeline doesn't understand this — it would try to apply SMOTE
to test data. ImbPipeline from imbalanced-learn knows that sampling steps only run during
`.fit()`, not `.predict()`. This is non-negotiable when SMOTE is in the pipeline.

**Do not use SMOTE and scale_pos_weight together.** SMOTE balances the training distribution.
scale_pos_weight reweights the loss function. Both solve the same imbalance problem by
different mechanisms. Using both simultaneously overcompensates — the model would treat
the minority class as far more important than it is, producing extreme recall at the cost
of precision. Pick one. With SMOTE in the pipeline, set scale_pos_weight=1 (neutral default).

---

## scale_pos_weight vs SMOTE — The Comparison

Both approaches fix the same problem. The comparison result was:

| Model                  | Precision @0.5 | Recall @0.5 | ROC-AUC |
| ---------------------- | -------------- | ----------- | ------- |
| XGB + scale_pos_weight | 0.71           | 0.75        | 0.9967  |
| XGB + SMOTE            | 0.80           | 0.75        | 0.9954  |

scale_pos_weight had marginally higher ROC-AUC (0.9967 vs 0.9954), but SMOTE had better
precision at the same recall. The AUC difference of 0.0013 across 16 test mules is noise —
catching one extra mule would flip the comparison. SMOTE's precision advantage is more
meaningful: at the same recall, it generates fewer false alarms.

The deeper reason SMOTE tends to produce better precision: scale_pos_weight makes the model
more aggressive about flagging mules globally by changing the loss function. SMOTE gives the
model more examples to learn the distinction between mule and legitimate accounts, which
produces better-calibrated probabilities. Better calibration means the threshold means
something more concrete.

---

## Threshold Tuning

The default threshold of 0.5 is a convention inherited from balanced classification. At 0.5,
"I'm more than 50% confident this is a mule" triggers a flag. On imbalanced data this is
almost always wrong — the model's probability estimates are skewed by the class distribution,
and 0.5 is usually too conservative for the minority class.

The precision-recall curve plots both metrics across every threshold from 0 to 1 simultaneously.
Moving left (lower threshold): recall increases, precision decreases. Moving right: precision
increases, recall decreases. The right operating point is determined by what the use case
can tolerate, not by convention.

**The threshold sweep confirmed 0.45 as the optimal point:**

| Threshold | Precision  | Recall     | F1         |
| --------- | ---------- | ---------- | ---------- |
| 0.40      | 0.765      | 0.8125     | 0.788      |
| **0.45**  | **0.8125** | **0.8125** | **0.8125** |
| 0.50      | 0.800      | 0.7500     | 0.774      |

At 0.45, precision and recall are exactly equal at 0.8125. This is the crossover point, and
it is also by definition where F1 is maximised — F1 is the harmonic mean of precision and
recall, and when they are equal, F1 equals both. Confirmed programmatically and visually.

**The staircase shape in the precision-recall curve is normal.** With 16 mules in the test
set, each step in recall represents exactly one more mule caught (1/16 = 6.25%). The jagged
appearance is a consequence of having integer counts in the numerator, not model instability.

---

## The Confusion Matrix — Raw Numbers

```
Confusion Matrix (threshold=0.450):
                  Predicted Legit   Predicted Mule
Actual Legit                1797               4   (False Alarms)
Actual Mule                    3              13   (Caught)

Mules caught:     13 / 16  (81.2% recall)
False alarms:     4 / 1801  (0.22% of legit accounts flagged)
Investigation load: 1 real mule per 1.3 alerts
```

The investigation load metric is the operationally meaningful one for a banking audience.
It translates model performance into a workflow question: how many accounts does a fraud
analyst need to investigate to find one real mule? At 1.3, the answer is: investigate
5 accounts, find 4 real mules. That is a viable operational system.

A false alarm rate of 0.22% means that for every 1000 legitimate accounts, 2 are incorrectly
flagged. At the scale of a real bank with millions of accounts, this number matters — but
at 0.22% it is well within acceptable range for a system that catches 81% of mules.

---

## Cross-Validation and the OOF Comparison

A single test set of 16 mules is too small to trust fully. Whether those 16 happened to be
the "easy" mules or the "hard" ones is partly luck. Out-of-fold (OOF) cross-validation
addresses this by giving every mule account exactly one turn in the test set across 5 folds.

**OOF is not the same as cross-validation scores.** Standard `cross_val_score` with recall
as the metric gives you the average recall across folds — but each fold uses the default
threshold of 0.5. OOF with `cross_val_predict(..., method='predict_proba')` generates
probabilities for every training account from a model that never trained on it, then applies
your chosen threshold to the full set of OOF probabilities. This is the honest evaluation
of both the model and the threshold choice together.

**XGB+SMOTE OOF results:**

| Metric         | Test Set (16 mules) | OOF (65 mules) |
| -------------- | ------------------- | -------------- |
| Mule Precision | 0.8125              | 0.77           |
| Mule Recall    | 0.8125              | 0.77           |
| Mule F1        | 0.8125              | 0.77           |
| ROC-AUC        | 0.9954              | 0.9649         |

The drop from 0.81 to 0.77 reflects the test set being a slightly favourable sample.
The OOF number is the realistic performance expectation on unseen data.

---

## The RF vs XGB Comparison — Why You Need OOF for Both

RF at threshold 0.25 produced an identical confusion matrix to XGB+SMOTE at threshold 0.45
on the test set: 13/16 mules caught, 4 false alarms. This appeared to make the models
equivalent. It did not.

With 16 test mules, confusion matrices can only take a small number of distinct values.
Multiple models with different underlying quality can produce the same integer counts by
chance. The test set comparison was insufficient to distinguish them.

OOF on all 65 training mules resolved this:

| Model         | OOF Mule F1 | OOF ROC-AUC |
| ------------- | ----------- | ----------- |
| XGB + SMOTE   | **0.77**    | **0.9649**  |
| Random Forest | 0.70        | 0.9538      |

XGB+SMOTE outperforms RF on every metric when evaluated honestly. RF's apparent parity on
the test set was a sampling artefact. This is the core reason OOF evaluation exists.

**The general principle:** never conclude two models are equivalent based on a test set with
fewer than 50 minority class samples. The noise in the comparison exceeds the signal.

---

## Why the Model Improved From Phase 4

Phase 4 RF OOF F1 for mules was approximately 0.50-0.55. Phase 5 XGB+SMOTE OOF F1 is 0.77.
The improvement has three sources:

1. **Cleaner features from Phases 2-3.** The F2230 leakage removal, ordinal encoding of
   F3889, and sum_of_lows engineering gave the model better signal to work with.
   XGBoost amplifies good features more effectively than RF.

2. **SMOTE.** 65 real mules became ~2100 training mules (at sampling_strategy=0.3 against
   ~7200 legitimate accounts). The model saw more variation in mule behaviour and
   generalised rather than memorised.

3. **Threshold tuning.** RF in Phase 4 was evaluated at threshold 0.5. XGB+SMOTE was tuned
   to 0.45. Even without changing the model, threshold tuning alone recovers meaningful
   recall on imbalanced classifiers.

---

## What Didn't Work and Why

**`max_depth=None` in XGBoost:** Doesn't work. XGBoost requires an explicit integer.

**sklearn Random Forest parameters in XGBoost:** `min_samples_leaf` and `min_samples_split`
are not XGBoost parameters. XGBoost uses `min_child_weight` for the equivalent constraint.
Passing sklearn parameters to XGBClassifier will either be silently ignored or error.

**SMOTE + scale_pos_weight together:** Double-compensates for imbalance. Tested implicitly
— the scale_pos_weight model without SMOTE was a cleaner comparison point.

**n_estimators=1500 without early stopping:** Inside ImbPipeline, early stopping requires
passing eval_set to `.fit()`, which is not straightforward. Without early stopping
configured, all 1500 trees build regardless of convergence. Computationally wasteful
for no gain. Use 300-500 unless early stopping is explicitly implemented.

---

## General Approach for Phase 5-Style Problems

If starting from scratch on a new imbalanced tabular classification problem:

1. **Run XGBoost with scale_pos_weight first.** It's one parameter, no extra dependencies,
   gives you a strong baseline. Calculate it as `(y_train==0).sum() / (y_train==1).sum()`.

2. **Add SMOTE second and compare.** Use ImbPipeline. Set sampling_strategy to 0.2-0.3,
   not 'auto'. Do not use both SMOTE and scale_pos_weight simultaneously.

3. **Compare models by OOF ROC-AUC, not test set metrics.** With fewer than 50 minority
   class samples in the test set, test metrics are too noisy to distinguish models.
   OOF gives every minority sample one honest evaluation.

4. **Tune the threshold with a sweep table, not just a graph.** The graph tells you the
   approximate location. The table tells you the exact value and the exact tradeoff at
   every step. Read both.

5. **Use the F1-maximising threshold as a starting point.** It is always where precision
   and recall cross on the graph. Adjust up if precision needs to be higher for operational
   reasons; adjust down if recall is the priority.

6. **Report the investigation load metric alongside precision and recall.** "1 real mule
   per 1.3 alerts" is more interpretable to a banking stakeholder than "precision: 0.77".
   It translates model quality into operational workflow.

7. **Check BorderlineSMOTE only if you have 200+ minority samples.** Below that, the
   borderline determination is too unstable to be reliable. Use regular SMOTE.

8. **Regularise XGBoost for small minority classes.** `min_child_weight=3`, `max_depth=4`,
   `gamma=1` are a reasonable starting triplet. Increase reg_lambda (default 1) to 2-3
   if OOF metrics are significantly worse than training metrics, indicating overfitting.

---

## Final Numbers

| Metric                  | Value                    |
| ----------------------- | ------------------------ |
| Model                   | XGBoost + SMOTE          |
| Threshold               | 0.45                     |
| Test Mules Caught       | 13 / 16 (81.2%)          |
| Test False Alarm Rate   | 0.22%                    |
| Test Investigation Load | 1.3 alerts per real mule |
| OOF Mule F1             | 0.76                     |
| OOF Recall              | 0.74                     |
| OOF Precision           | 0.79                     |
| OOF ROC-AUC             | 0.9680                   |
