# Phase 3 — What We Actually Did, Why, and How to Think About It

---

## What Phase 3 Was Really About

Phase 3 had a single purpose: **take the 2000+ raw features you had after Phase 2 and turn them into ~50 features that carry as much signal as possible.** That breaks into two sub-problems:

1. **Feature Engineering** — create new columns that make existing signal more obvious to a model.
2. **Feature Selection** — use a model to rank everything and keep only what matters.

The order matters. You engineer first, then select — so the new features you create compete fairly against the originals. If you selected first and engineered after, you'd be building new features from a restricted subset and never testing whether they beat what you threw away.

---

## Part 1 — Feature Engineering: The Theory

### Why a Model Needs Help Seeing Patterns

A machine learning model — especially a tree-based one like Random Forest or XGBoost — sees each account as a row of numbers. It finds patterns by splitting: "if column X is below some threshold, go left; otherwise go right." It repeats this hundreds of times across hundreds of trees.

The problem: some patterns only exist **across** columns, not within any single one. A mule account might have *high inflow* and *high outflow* — each of those values individually looks normal, but the *relationship* between them is abnormal. The model can technically discover multi-column patterns (a tree can split on column A first, then on column B), but it's much harder — it requires the right split on the right column to happen early in the tree, or the signal gets buried.

Feature engineering pre-computes these relationships. Instead of asking the model to discover that `F670 / F2678` is extreme for mules, you hand it a single column where that ratio is already calculated. One split instead of two. The model's job gets easier, and it performs better.

**General principle:** Feature engineering is the act of encoding your domain knowledge into the data format the model expects. You're translating "I think mule accounts have disproportionate activity ratios" into a column the model can directly split on.

### The Four Types of Engineered Features We Created

#### 1. Ratio Features (Imbalance Detection)

```python
X['F531/F321'] = X['F531'] / (X['F321'] + 1e-6)
```

**What this captures:** When two features are correlated (F531 and F321 had r = 0.49), they're partially measuring the same underlying thing — probably two types of transaction activity. Their *ratio* captures the imbalance between them. An account where F531 is much larger than F321 behaves differently from one where they're roughly equal.

**Why the `+ 1e-6`:** Division by zero produces `NaN` or `inf`, both of which break models. Adding a negligible constant (0.000001) to the denominator prevents this without meaningfully changing the result. This is standard practice for any ratio feature. The constant is small enough to be irrelevant for any non-zero denominator, and for zero denominators it turns `x / 0` into `x / 0.000001`, which gives a very large number rather than infinity — still interpretable.

**When to use ratios generally:** Whenever you have two features that either (a) measure related things, or (b) one of which is high for the target class while the other is low. Ratios amplify divergence: if mules have high F670 and low F2678, then `F670 / F2678` will be dramatically higher for mules than for anyone else.

#### 2. Interaction Features (Signal Amplification)

```python
X['F2678*F2122'] = X['F2678'] * X['F2122']
```

**What this captures:** If both F2678 and F2122 are low for mules, their product is *very* low — low × low = tiny. For legitimate accounts where both are moderate, the product stays moderate. The gap between classes gets amplified.

**The mathematical intuition:** Think of it geometrically. If a mule sits at (F2678=2, F2122=0.005), its product is 0.01. A legit account at (F2678=252, F2122=0.046) has a product of 11.59. The multiplicative feature turns a 126x difference on one axis and a 9x difference on the other into a **1159x** difference in one column. That's an exponentially stronger signal for the model to split on.

**When to use interactions generally:** When multiple features point in the same direction for the target class (all low, or all high). Multiplying them together compounds the signal. Be cautious with features pointing in opposite directions — the product might cancel out or be hard to interpret.

#### 3. Aggregate Features (Composite Signals)

```python
X['sum_of_lows'] = X['F2678'] + X['F2122'] + X['F2956'] + X['F2737'] + X['F1692']
```

**What this captures:** Five features were all moderately to strongly lower for mules. Individually, each one's signal might be noisy — maybe F2737 is low for some legitimate accounts too. But the *sum* of all five captures the overall "activity level." A mule that's low on all five will have a very low sum. A legit account that happens to be low on one or two will still have a moderate sum from the other three.

**Why this is more robust than individual features:** It's the same principle as taking an average of multiple measurements. One measurement might be noisy; five measurements averaged together are more reliable. The noise cancels, the signal accumulates.

**When to use aggregates generally:** When you have a group of features that all represent the same underlying concept (here: "account activity level"). Summing or averaging them creates a single composite score that's more stable than any individual component.

#### 4. Z-Score Features (Anomaly Detection)

```python
for col in ['F2678', 'F2122', 'F670', 'F2956', 'F2737']:
    col_mean = X[col].mean()
    col_std = X[col].std()
    if col_std > 0:
        X[f"{col}_zscore"] = (X[col] - col_mean) / col_std
```

**What this captures:** A z-score answers: "how many standard deviations away from the average is this account?" A z-score of -2.0 means "this account's value is 2 standard deviations below the mean" — it's unusual. Mule accounts, if they systematically differ from legitimate ones, should cluster at unusual z-scores.

**Why it largely failed here:** The z-score is computed across all 9,082 accounts together. Since 9,001 are legitimate, the mean and standard deviation are essentially the *legitimate account* statistics. The 81 mules barely influence the computation. For most features, the mule median z-score ended up identical to the legit median z-score, because both classes have the same "bulk" — the signal in z-scores lives in the tails, not the middle. With only 81 mules, the tails don't have enough mass to shift the median.

**The lesson:** Z-scores work best when both classes are reasonably balanced, or when the minority class is *extremely* far from the majority. In moderate-imbalance fraud detection with subtle differences, they tend to be ineffective. The one exception was `F2956_zscore` (ratio 1.377), where mules were consistently further into the negative tail.

**When to use z-scores generally:** On balanced datasets, or as an input to outlier detection methods. For imbalanced classification, they're usually the weakest engineered features. Don't bother unless you have a specific reason.

---

### The Philosophical Point About Feature Engineering

Feature engineering isn't about blindly creating combinations. It's about translating hypotheses into columns:

- "Mules receive money and pass it on immediately" → ratio of inflow to outflow features
- "Mules have low overall activity" → sum of activity features
- "Mules are unusual on specific dimensions" → z-scores
- "The combined weakness on multiple features is more diagnostic than any single feature" → interaction products

Every engineered feature is a *bet* that a certain relationship matters. Some bets pay off, some don't. You won't know until you test them — which is what Part 2 does.

---

## The Diagnostic Interlude: Validating Engineered Features

Before running feature selection, we ran a diagnostic to check whether the engineered features actually carried signal. This is where **choosing the right summary statistic** became critical.

### Why Median Failed on Sparse Features

For features like `F2678*F2122`, 68 out of 81 mules had value 0, and 5,974 out of 9,001 legit accounts also had value 0. The median for both classes was 0. The median ratio was 0/0 — undefined.

But the *means* were completely different: mule mean -0.0015 vs legit mean 0.2935. Legit accounts were 200x higher on average. The signal was real — the median just couldn't see it.

**The general lesson:** For **sparse features** (where most values are zero), median is useless. Mean, zero-percentage, and tail statistics are the right tools. For **dense features** with extreme outliers, median is more trustworthy than mean. You need both — and you need to know when each one lies.

### The Three Diagnostic Lenses

We ended up needing three independent checks to properly assess each engineered feature:

| Metric | What it captures | When to trust it |
|--------|-----------------|------------------|
| **Median ratio** (mule/legit) | Typical account difference | Dense features without extreme outliers |
| **Mean ratio** (mule/legit) | Overall difference including outliers | Always informative, but easily distorted by extreme values |
| **Zero-percentage gap** between classes | Whether the classes have different sparsity patterns | Sparse features where median/mean are misleading |

A feature qualifies as signal if **any one** of the three passes its threshold:
- Median ratio: < 0.70 or > 1.40
- Mean ratio: < 0.50 or > 2.00
- Zero-percentage gap: > 10% between classes

The OR logic is important. Requiring all three simultaneously (AND logic) would kill every sparse feature — because sparse features always have median = 0 for both classes. You'd be systematically discarding features that only show signal through their sparsity pattern.

### The "Seeing Things" Intuition That Was Actually Correct

> *"it does look like with a special combination of features mules can be pointed out. just my gut feeling, kinda like a DBMS combining multiple foreign keys to get that one record."*

This intuition is the entire mathematical reason tree-based models exist. No single feature definitively identifies a mule. But a decision tree says: "if F2678 is low AND F670 is high AND F3898 < 3 → mule." That's a composite key. Each column alone isn't unique, but together they isolate exactly the right records.

This is also why feature engineering is fundamentally different from feature selection. Engineering creates *new* composite keys. Selection identifies which keys (original or engineered) the model actually uses.

---

## Part 2 — Feature Selection: The Theory

### What Feature Importance Actually Measures

A Random Forest is an ensemble of decision trees. Each tree is built by:
1. Taking a random sample of rows (with replacement — called "bagging")
2. At each node, considering a random subset of columns (√n_features by default)
3. Picking the column and split point that best separates mules from legit accounts at that node
4. Repeating until stopping conditions are met

After training, **feature importance** is computed by tracking, for each feature: how many times was it used for a split across all trees, and how much did each of those splits improve the separation (measured by Gini impurity reduction or information gain)?

The resulting number — normalized to sum to 1.0 across all features — is a rough ranking of "how much did the model rely on this feature." High importance means the model used it frequently and productively. Low importance means it was rarely useful.

**Critical caveat:** Feature importance from a single Random Forest is a *rough estimate*, not ground truth. It's biased in several ways:
- Features with more unique values tend to get higher importance (more possible split points)
- Correlated features split importance between them (both appear moderate even if only one is truly useful)
- Fully grown trees can overfit to the minority class, making importances partially reflect memorization

We used the RF for feature selection specifically — not as the final model. The XGBoost in later phases is the truth test.

### Why We Used These Specific RF Hyperparameters

```python
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=6,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**`n_estimators=500`** — More trees = more stable importance estimates. With 100 trees, the rankings might shift if you change the random seed. With 500, they stabilize. Beyond ~500, diminishing returns.

**`max_depth=None`** — Let trees grow fully. Shallow trees (max_depth=5) are good for fast training but miss complex interactions. Since we want the most informative importance ranking possible and don't care about training time, let them grow.

**`min_samples_leaf=3` and `min_samples_split=6`** — The brakes. Without these, a fully grown tree would create leaves with a single account in them — memorizing individual mule accounts rather than learning general patterns. Requiring at least 3 accounts per leaf and 6 to attempt a split forces the tree to generalize. With only 81 mules, these constraints are critical.

**`class_weight='balanced'`** — The 81 vs 9,001 imbalance means that without weighting, the RF would almost never see a mule in its random samples. `balanced` automatically upweights the minority class proportionally, so each mule account contributes ~111x more to the loss than each legit account. This ensures the model actually learns to distinguish mules, rather than just predicting "legit" for everything and getting 99.1% accuracy.

**`max_features='sqrt'`** — At each split, only consider √2032 ≈ 45 random features. This decorrelates the trees — if every tree considered all features, they'd all make similar splits and give similar importances. Randomizing forces diversity, making the ensemble more robust.

**`random_state=42`** — Reproducibility. Same seed = same results every run. Essential for debugging.

**`n_jobs=-1`** — Use all CPU cores. Pure performance optimization, no statistical effect.

### What Healthy Feature Importances Look Like

After fixing the leakage bugs, the top 30 importances were:

```
F3898    0.016217
F3806    0.014054
F3801    0.013299
F3807    0.013250
...
F1273    0.005468
```

**Key observations:**

1. **No single feature dominates.** The highest importance is 0.016 — roughly 3x the lowest in the top 30 (0.005). This is healthy. When one feature has importance 0.5+ and everything else is <0.01, that's almost certainly leakage.

2. **The importances are smoothly distributed.** No sharp cliff between "important" and "not important." This means the model is genuinely distributing its decisions across many features — exactly what you want for a robust classifier.

3. **None of the hint features appeared in the top 30.** Surprising, but not alarming. The hint features were chosen by the hackathon organizers as "potentially useful" — not as guaranteed top predictors. The RF found 50 other features that, individually, were more useful for splitting. The hint features might still contribute in the final model through interactions.

### Why the Engineered Features Didn't Make the Top 50

None of the 15 engineered features appeared in the top 50. This is actually informative:

**What it means:** The raw features already captured the patterns that the engineered features were trying to express. For example, `F2678*F2122` was trying to amplify the "both are low for mules" signal — but the RF could already find that by splitting on F2678 first and then F2122 in a subtree. The product didn't give it anything it couldn't already do.

**What it does NOT mean:** That the engineering was wasted. The engineering process confirmed our understanding of the data, tested hypotheses, and verified that the patterns we saw in Phase 2 were real. It also served as a sanity check — if an engineered feature had dramatically outperformed its component features, that would have been strong validation.

**The one exception — `sum_of_lows`:** This showed genuine signal in the diagnostic (median ratio 0.648) and was force-included in the final 50+1 feature set. The RF might not have ranked it in the top 50, but the diagnostic showed it carries real information that might complement the other selected features.

**General lesson about feature engineering:** Most engineered features don't survive selection. A 20% hit rate (3 out of 15 making the cut) is considered good in practice. The value of engineering isn't in the features that survive — it's in the understanding you build and the occasional breakthrough feature that captures something no raw feature can.

---

## The Leakage Bugs: What Actually Went Wrong

Two bugs nearly invalidated Phase 3 entirely. Both are embarrassingly common and worth understanding deeply.

### Bug 1 — The Target Column in the Feature Set

```python
X_final = pd.read_csv('final_dataset_with_target.csv')
y = X_final['target']
X_final.drop(columns=['target'])  # BUG: returns a new DataFrame, doesn't modify in place
```

Pandas' `.drop()` method returns a **new** DataFrame. It does not modify the original. The line above computes a new DataFrame without the target column, then immediately discards it — the variable `X_final` still contains the target.

The RF then trained on features *including the correct answer*. The target column had importance 0.063 — second highest. The model was literally using the label to predict the label. Every evaluation metric would be a lie.

**The fix:** Assign the result back.

```python
X_final = X_final.drop(columns=['target'])
```

**Or use `inplace=True`:**

```python
X_final.drop(columns=['target'], inplace=True)
```

**The general lesson:** In pandas, most transformations are non-destructive by default — they return a new object. This is by design (to prevent accidental data loss), but it's a constant source of bugs when you forget to capture the return value. Get in the habit of adding `assert` checks after any operation that's supposed to modify your data:

```python
assert 'target' not in X_final.columns
```

### Bug 2 — The Row Index Saved as a Feature

When saving the CSV earlier, `index=False` was omitted. Pandas wrote the row numbers (0, 1, 2... 9081) as an unnamed column, which got loaded as `IndexSS`. If the data happened to be sorted in any way that clustered mule accounts together (e.g., mules were all at the end of the dataset), the row index would correlate with the target. The RF exploited this — IndexSS had the highest importance at 0.066.

**Why this is leakage:** The row index carries information about data ordering, which is an artifact of data collection — not a genuine property of the accounts. A model trained on this would learn "accounts near row 9000 are mules" — completely useless on new data where row numbers are arbitrary.

**The fix:** Always save with `index=False`:

```python
df.to_csv('output.csv', index=False)
```

And always check what columns exist after loading:

```python
print(X_final.columns.tolist())
```

---

## The F38xx Leakage Investigation

Ten of the top 30 features were in the F3800 range — F3898, F3806, F3801, F3807, etc. The target itself was F3924, also in the 3800s. This raised a red flag: were these features somehow derived from the target? Were they metadata assigned *after* fraud detection?

We investigated by looking at value distributions grouped by target:

**F3898** turned out to be a count feature (values 0–57). The class split was telling:
- 53% of legit accounts had value 3; only 2.5% of mules did
- 48% of mules had value 0; only 31% of legits did

This is real, interpretable signal. Whatever F3898 counts (probably banking products, linked accounts, or relationship depth), mule accounts rarely reach 3. They're created for a specific purpose and used immediately — no time to accumulate a banking relationship.

**F3806 and F3801** were continuous monetary features. Both classes had unique values (normal for balances), no suspicious patterns.

**The general approach to leakage investigation:**
1. **Check if values are suspiciously clean.** Binary flags (0/1 only) that perfectly separate classes are almost always assigned post-hoc.
2. **Check if the feature makes domain sense.** Can you tell a story about *why* this feature would differ between mules and legit accounts that doesn't involve knowing the label first?
3. **Check the value_counts grouped by target.** If one class has a completely different distribution that can't be explained by the underlying phenomenon, be suspicious.

---

## The General Approach: How to Do Phase 3 on Any Problem

### Step 1: Identify Your Signal Features

From your EDA (Phase 2), you should have a ranked list of features that show statistical differences between classes. Group them:

- **Features where the target class is higher** (e.g., F670 for mules)
- **Features where the target class is lower** (e.g., F2678, F2122 for mules)
- **Correlated pairs** that might carry redundant information

### Step 2: Engineer Based on Hypotheses

For each group, ask: "What relationship between these features would be most diagnostic?"

| Pattern you observe | Engineering strategy |
|---------------------|---------------------|
| Two features both high or both low for target class | Multiply them (amplifies the signal) |
| One feature high, another low for target class | Divide them (ratio captures the divergence) |
| Two features are correlated | Divide them (ratio captures the imbalance between them) |
| Multiple features all point the same direction | Sum or average them (composite signal, more robust) |
| You want to flag "unusual" accounts | Z-score, percentile rank, or deviation from median |

### Step 3: Validate Before Selecting

Don't trust your engineering blindly. Run diagnostics:
- For dense features: median ratio between classes
- For sparse features: mean ratio and zero-percentage gap
- Drop anything that shows no signal on any metric

### Step 4: Train a Quick Model for Feature Importance

Use a Random Forest with:
- Enough trees for stable importances (~100–500)
- Controls against overfitting (`min_samples_leaf`, `min_samples_split`)
- Class weighting for imbalanced data
- All available features (so engineered and original compete fairly)

### Step 5: Select Top N Features

Take the top 50–100 by importance. Force-include any engineered features that showed strong diagnostic signal but didn't make the RF cut (the RF can be wrong — it's a rough ranker, not an oracle).

### Step 6: Sanity Check the Final Set

For every feature in your final set:
- Is it a legitimate account property, or an artifact (index, timestamp, metadata)?
- Can you explain *why* it would differ between classes without circular reasoning?
- Does its value distribution make domain sense?

If anything fails these checks, investigate before proceeding. One leaky feature can invalidate an entire model.

---

## What You Should Take Away From Phase 3

1. **Feature engineering is hypothesis-driven.** You're not randomly combining columns — you're translating domain intuitions into computable features. The combination types (ratios, products, sums, z-scores) are tools, not recipes.

2. **Most engineered features will fail.** That's normal. The value is in the few that succeed and in the understanding you build along the way.

3. **The right summary statistic depends on the data distribution.** Median for dense, mean for sparse, zero-percentage for binary-like. Using the wrong one gives you wrong conclusions.

4. **Feature selection with a Random Forest is a rough filter, not a final answer.** It's biased toward features with many unique values, it splits importance among correlated features, and it can overfit to the minority class. Use it to reduce 2000 features to 50, then let your final model (XGBoost) decide which of those 50 actually matter.

5. **Leakage is always the biggest risk.** At every stage — saving CSVs, loading data, engineering features, selecting features — check that the target isn't leaking into the feature set. The `assert` pattern is cheap insurance.

6. **Pandas `.drop()` doesn't modify in place.** This will bite you again if you forget. Assign the result back, or use `inplace=True`. Always verify with `assert`.

7. **The composite key intuition is real.** No single feature identifies a mule. The model learns multi-feature "keys" — combinations of thresholds across many features that together isolate the target class. Feature engineering is about making those keys easier to find.
