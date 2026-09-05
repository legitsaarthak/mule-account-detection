# Phase 4 — What We Actually Did and Why

---

## The Goal of Phase 4

Phase 4 had one job: establish baselines. Not build the best model — establish the worst acceptable
model, so you have something to beat. This is a discipline thing. If you skip straight to XGBoost
and tune it aggressively, you have no reference point. You don't know if your 0.85 recall is good
because of XGBoost specifically, or because the data was already clean from Phase 3, or because
any model would have gotten 0.85 on this dataset. Baselines answer that question.

The two models chosen — Logistic Regression and Random Forest — weren't chosen because they're
the best. They were chosen because they represent two completely different philosophies of
classification, which makes the comparison informative.

---

## The Train/Test Split — And Why Stratification Is Non-Negotiable Here

Before training anything, you split the data: 80% for training, 20% for testing. The model
learns on the training set and gets evaluated on the test set — data it has never seen. This
prevents you from measuring how well the model memorized the training data instead of how well
it generalizes to new data.

The critical detail for imbalanced problems: **stratification**.

A random split doesn't guarantee proportional representation. With 81 mules in 9082 accounts,
a bad random split could put 70 mules in training and 11 in test — or in a worst case, even
fewer. When you then evaluate on 11 mules, your recall numbers are wildly unstable. Catching
one extra mule swings recall by 9%.

`stratify=y` forces both the training and test splits to maintain the same ~0.89% mule ratio.
It's not optional on this dataset — it's the difference between a meaningful evaluation and a
coin flip dressed up as science.

---

## Why a Pipeline

A Pipeline chains preprocessing steps and the model into a single object. This isn't just
code hygiene — it prevents a specific class of bug called **data leakage**.

Here's the problem it solves: StandardScaler needs to learn the mean and standard deviation
of each feature to scale them. If you fit the scaler on the full dataset before splitting,
the scaler has seen the test data. It knows the test set's distribution. When you later
evaluate on the test set, the model is technically operating on data it's already been
exposed to — even if indirectly.

The correct procedure: fit the scaler only on training data, then apply that same
pre-learned scaling to the test data. Pipeline enforces this automatically. When you call
`pipeline.fit(X_train, y_train)`, the scaler fits on X_train only. When you call
`pipeline.predict(X_test)`, it applies the training-set scaling to X_test without refitting.

You already encountered leakage twice in Phase 3 (F2230 and the row index bug). Pipeline
closes the door on this particular version of it so you don't have to think about it again.

---

## Logistic Regression — What It's Actually Doing

Logistic Regression finds a hyperplane — a flat dividing surface — through feature space
that best separates mules from legitimate accounts. In two dimensions you'd visualize this
as a straight line. With 50 features it's a 50-dimensional flat surface, but the concept
is identical.

Every feature gets a weight. Positive weight: higher values push toward "mule". Negative
weight: higher values push toward "legitimate". The model learns these weights by adjusting
them until the dividing surface correctly classifies as many training examples as possible.

**Why it needs scaling:** The hyperplane is found by measuring distances and gradients in
feature space. A feature measured in millions (like some of the F38xx range) would dominate
a feature measured in single digits (like F3889's 1-7 ordinal encoding), because the
algorithm treats them as existing in the same space. StandardScaler puts all features on
the same footing — mean 0, standard deviation 1 — so no feature dominates by accident
of its units.

**Why it's the first baseline:** LR makes a strong assumption — that the boundary between
classes is approximately linear. In most real-world fraud problems, it isn't. So LR gives
you the floor: what you get when your model is almost certainly too simple. Anything that
can't beat LR isn't worth using.

**`class_weight='balanced'`:** LR optimises for overall correct predictions. With 9001
legitimate and 81 mule accounts, predicting "legitimate" for everything gives 99.1% accuracy
and zero fraud caught. `class_weight='balanced'` tells the optimiser: when you misclassify
a mule, treat that as a much heavier error than misclassifying a legitimate account.
Specifically, it weights each class inversely proportional to its frequency — mule errors
get penalised roughly 111x more than legitimate errors (9001/81 ≈ 111). This forces the
model to take minority class errors seriously.

---

## Random Forest — What It's Actually Doing

Random Forest builds hundreds of decision trees, each asking a sequence of yes/no questions
about the features to arrive at a prediction. Each tree is trained on a random bootstrap
sample of the training data (sampling with replacement — some rows appear multiple times,
some not at all). Each split in each tree considers only a random subset of features
(√50 ≈ 7 features per split). The final prediction is the majority vote across all trees.

The randomness serves a specific purpose: it ensures the trees are different from each other.
If you built 700 identical trees on the same data, voting wouldn't help — they'd all make
the same mistakes. By randomising the training samples and features seen at each split,
each tree has different blind spots. Their errors don't correlate. When you average out
uncorrelated errors, they cancel. This is the core mechanic that makes ensembles more
robust than any individual tree.

**Why no scaling:** Trees split on thresholds — "is F2678 < 0.42?". They don't compute
distances or gradients. Whether F2678 is measured in original units or scaled units, the
threshold just moves proportionally. The model is identical. Scaling is irrelevant.

**`class_weight='balanced_subsample'` vs `'balanced'`:** Regular `'balanced'` computes
class weights once from the full training set and applies them to every tree. 
`'balanced_subsample'` recomputes class weights from each tree's bootstrap sample. Since
each sample has a different mule-to-legitimate ratio by chance, this creates additional
diversity between trees — each tree is calibrated to its own slice of the data. On
imbalanced datasets with bootstrap sampling, this is more principled than a single global
weight.

**Why `min_samples_leaf=3`:** With only ~65 mule accounts in the training set, trees
allowed to grow to individual-sample leaves will memorize specific mules rather than
learning generalizable patterns. `min_samples_leaf=3` forces every leaf to represent
at least 3 accounts. The model is compelled to find patterns that hold across multiple
mules, not just fit to one.

This is a regularisation decision specific to extreme imbalance. On a balanced dataset
with thousands of examples per class, `min_samples_leaf=1` is fine. Here, with 65
positive examples, it's overfitting bait.

---

## Reading the Results — The Precision-Recall Tradeoff

This is the most important conceptual thing Phase 4 teaches.

The results looked like Logistic Regression outperformed Random Forest:

```
                 Precision   Recall    F1      ROC-AUC
LR               0.0505      0.9375    0.0958  0.9672
RF               0.9000      0.5625    0.6923  0.9918
```

This reading is wrong. LR didn't outperform RF. They made completely different tradeoffs.

**What LR actually did:** Flagged ~297 accounts as mules. Caught 15 out of 16 real mules.
But 282 of those 297 flagged accounts were legitimate. Precision of 5% means 95% of its
alerts are false alarms.

**What RF actually did:** Flagged 10 accounts as mules. Caught 9 out of 16 real mules.
Only 1 of those 10 was a legitimate account. Precision of 90% means almost every alert
is real.

Neither is "better" in isolation — it depends entirely on what you're optimising for.
In fraud detection, missing a real mule (False Negative) is generally worse than
investigating a legitimate account (False Positive). So recall matters more. But a
model with 5% precision is generating 282 false investigations per 15 catches — that's
operationally unacceptable at scale even if recall is high.

The goal isn't maximum recall. It's **maximum recall subject to precision being
operationally viable**. Phase 5 is specifically designed to find this middle ground.

**Why ROC-AUC is the honest comparison:** ROC-AUC measures how well the model *ranks*
accounts from most to least suspicious across every possible threshold — not just at the
default cutoff of 0.5. It answers: "if I randomly pick one mule and one legitimate account,
what's the probability my model assigns the mule a higher suspicion score?" RF at 0.9918
vs LR at 0.9672 means RF is the better ranker by a meaningful margin. The default threshold
of 0.5 just happens to make RF look conservative — lower it and RF's recall rises while
maintaining far better precision than LR achieves.

The general rule: **on imbalanced classification problems, never compare models by recall
or precision alone. Compare by ROC-AUC first, then choose your threshold based on the
operational precision-recall tradeoff you can accept.**

---

## Cross-Validation — Why One Split Isn't Enough

The train/test split gives you one evaluation on one particular set of 16 mules. Whether
those 16 mules happen to be the "easy" ones or the "hard" ones is luck. Your recall
number is partly a function of which mules ended up in the test set, not just how good
your model is.

StratifiedKFold cross-validation splits the data 5 different ways, trains and evaluates
5 times, and averages the results. Each mule account ends up in the test set exactly once
across the 5 folds. The average performance is a much more honest estimate of how the
model generalizes.

The standard deviation matters as much as the mean:

```
RF recall: 0.5037 ± 0.1577
```

That ±0.1577 looks alarming — it means RF recall ranged roughly from 0.35 to 0.66 across
folds. But this is a math artifact of tiny numbers, not model instability. With ~16 mules
per test fold, catching one extra mule changes recall by 6.25%. The variance is inherent
to the problem size, not the model. Don't over-interpret it.

What the CV confirms: RF's `roc_auc: 0.9735 ± 0.0201` vs LR's `0.9330 ± 0.0207` — RF
consistently ranks accounts better across all 5 folds. The single-split result wasn't a
fluke.

---

## The ROC Curve — Threshold as a Design Decision

The ROC curve shows the tradeoff between True Positive Rate (recall) and False Positive
Rate (fraction of legitimate accounts falsely flagged) across every possible threshold from
0 to 1.

At threshold 1.0: the model flags nothing. Recall = 0, FPR = 0.
At threshold 0.0: the model flags everything. Recall = 1, FPR = 1.
Every point in between is a different operating point.

The "best" threshold isn't 0.5 — that's just the default. The right threshold is wherever
the precision-recall tradeoff is operationally acceptable for your use case. A bank with
a large fraud investigation team can tolerate a lower threshold (higher recall, more false
positives). A small team needs a higher threshold (fewer alerts, each one more reliable).

This is a business decision, not a data science one. Your model doesn't make this choice —
it just gives the bank the curve and lets them choose their operating point. In Phase 6,
when you present your results, showing this curve demonstrates that you understand the
difference between building a model and deploying one.

---

## What the Phase 4 Results Actually Mean

RF ROC-AUC of 0.9918 at the baseline stage — before SMOTE, before XGBoost, before any
tuning — is a strong result. The reason it's strong isn't luck. It's because of the work
done in Phases 2 and 3: catching the F2230 leakage, encoding F3889 ordinally, identifying
which hint features carry genuine signal, building sum_of_lows. Clean features produce
clean signal. The model's job becomes much easier when the inputs aren't contaminated.

Phase 5 has one specific goal: push RF's recall (0.5625) up significantly while keeping
precision acceptable. XGBoost is more powerful than RF on tabular data, and SMOTE gives
the model more mule examples to learn from during training. The combination should produce
a model that catches more mules without generating an operationally unacceptable false
alarm rate.

The baseline is set. The gap to close is clear. That's exactly what Phase 4 is for.

---

## General Approach for Imbalanced Binary Classification

If you were starting a new fraud-style problem from scratch, the Phase 4 mental checklist:

1. **Always stratify your train/test split.** On imbalanced data, this isn't optional.

2. **Always use a Pipeline.** Fit preprocessing only on training data. Never touch test
   data before evaluation.

3. **Start with Logistic Regression.** It gives you the floor. If LR is already strong,
   your features are doing the work. If LR is terrible, you know why you need something
   more powerful.

4. **Use `class_weight` from the start.** On any imbalanced problem, a model without
   class weighting will default to predicting the majority class. This is useless for
   fraud detection. Apply `'balanced'` or `'balanced_subsample'` before anything else.

5. **Never compare imbalanced classifiers by accuracy or by recall alone.** Compare by
   ROC-AUC. Then look at the precision-recall tradeoff to decide which threshold to use.

6. **Cross-validate, but interpret std with context.** High standard deviation on recall
   with tiny minority class counts is a dataset problem, not a model problem. Check the
   absolute number of minority samples per fold before panicking.

7. **Regularise aggressively when minority class is tiny.** `min_samples_leaf`, `max_depth`,
   `C` in LR — these prevent the model from memorizing the few positive examples it has
   instead of learning from them.

8. **The default threshold of 0.5 is arbitrary.** The right threshold is determined by
   operational constraints, not convention. Always present the ROC curve alongside point
   metrics so the end user can make an informed threshold decision.
