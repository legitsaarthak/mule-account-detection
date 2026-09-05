# Phase 2 — What We Actually Did and Why

---

## The Goal of Phase 2

Phase 2 had two jobs: finish cleaning the data so it's model-ready, and explore it visually to find which features carry signal. The cleaning part is mechanical. The exploration part is where most of the real thinking happens — and where most beginners go wrong by trusting the wrong tools.

---

## Part 1 — Cleaning

### Why Drop High-Null Columns

A column with 60% missing values isn't just incomplete — it's actively dangerous. Any pattern the model learns from that column is a pattern in _which accounts had data recorded_, not a pattern in the accounts themselves. That's a subtle form of bias. You'd be modelling the data collection process, not the fraud.

The 50% threshold is a rule of thumb, not a law. The real question is: at what point does a column have so little data that whatever it says is untrustworthy? 50% is a reasonable answer for most problems.

### Why Drop Near-Zero Variance Columns

A feature that's the same value for 99% of rows teaches the model nothing. The model learns by finding _differences_ between mule and legitimate accounts. If a column barely changes across all accounts, there's no difference to find. It's noise that wastes compute and can slightly degrade model quality by giving it useless splits to consider.

Variance is just a measure of how spread out values are. Zero variance = all identical. Near-zero = almost all identical.

### Why Impute With Median, Not Mean

The mean is the average. The median is the middle value. In fraud data — and most financial data — distributions are heavily skewed. One account might have transacted ₹50 crore while most accounts transact ₹10,000. The mean gets pulled toward that outlier. If you fill missing values with the mean, you're filling them with a number that isn't representative of a typical account.

The median ignores outliers. It always represents the "typical" account regardless of extremes. This is why median imputation is the default choice for financial and fraud datasets.

### Data Leakage — The F2230 Discovery

This was the most important find in Phase 2, and it's the kind of thing a course never teaches because courses use clean, pre-processed datasets.

F2230 contained month-year codes. All 9001 legitimate accounts had the value "Oct25". The 81 mule accounts were spread across other months. If you had kept this feature, the model would have learned: "if F2230 ≠ Oct25, it's probably a mule." The model would score near-perfectly on your dataset.

But in production, a new incoming account would have whatever the current month is — not Oct25. The feature would mean nothing. The model's performance would collapse completely on real data.

This is called **data leakage** — information in your training data that wouldn't be available in production, or that gives away the answer in a way that doesn't generalise. It's the single most common way that models with great evaluation scores fail in the real world.

The general rule: any feature that seems suspiciously powerful deserves a hard look. Ask why. If you can't explain the mechanism — why would this feature genuinely cause fraud risk? — be suspicious.

### Encoding — Why Different Categories Need Different Treatment

Not all categorical columns are equal. There are two fundamentally different types:

**Nominal categories** have no order. "Salaried", "self-employed", "student" — none of these is more than another. Label encoding (converting to integers 0, 1, 2...) technically implies an order that doesn't exist, but for tree-based models this doesn't matter. Trees split on "is value == X?" not on mathematical relationships between values. So label encoding is fine for XGBoost on nominal features.

**Ordinal categories** have a real, meaningful order. F3889 had values like L7D, L14D, L31D... G365D — account age buckets from youngest to oldest. Here order genuinely matters. An account older than 365 days is meaningfully different from one that's 7 days old, and the numbers assigned should reflect that. Manually mapping these to 1 through 7 in the correct order preserves that information.

The mistake most beginners make: applying label encoding to everything without asking whether order exists. On linear models, this causes real errors. On tree models, it's less damaging but still sloppy thinking.

---

## Part 2 — Exploration

### Why Histograms Failed Here

The instinct to plot distributions side by side is correct. The problem is that financial data almost always has extreme outliers — a small number of accounts with very large transaction values. These outliers stretch the x-axis so far that all the actual variation in the bulk of the data gets compressed into a thin sliver on the left.

The result: both mule and legitimate distributions look like identical flat lines, even when they're genuinely different. The histogram becomes useless as a diagnostic tool.

This is a scale problem, not a signal problem. The signal is still there — you just can't see it with this tool on this data.

The lesson: never trust a single visualisation. Always cross-check with numbers.

### Why the Ratio Table Was More Honest

The mean ratio (mule mean / legitimate mean) sidesteps the visualisation problem entirely. You're comparing a single number per class — no scale issues, no visual compression.

A ratio of 1.0 means mule and legitimate accounts have identical average values for that feature. The further from 1.0, the more the classes differ. This directly tells you which features carry discriminative signal — which is exactly what you need for feature engineering.

The caveat: the mean itself can be misleading when outliers are extreme, as F3836 showed. Legitimate accounts had a mean of −49 million but a median of 382K — because a handful of accounts with massive negative balances dragged the mean down. The median of 382K vs mule median of 338K tells a completely different story: nearly no difference between classes.

The general principle: **for skewed financial data, always check both mean and median before drawing conclusions from either one.**

### What Signal Actually Looks Like in This Data

The features that turned out to carry real signal (F2678, F2122, F2956, F2737, F1692, F670) shared a pattern: mule accounts consistently had either much lower or much higher values than legitimate accounts on average. The difference was real but subtle enough that visualisations couldn't show it.

This is normal for fraud detection. Fraudsters don't behave in radically alien ways — if they did, they'd be caught immediately by simple rules. They behave almost like normal accounts, with differences that only emerge statistically across many accounts. That's precisely why rule-based systems fail and why you need a model.

### What Correlation Tells You and What It Doesn't

Correlation measures the linear relationship between two features. A high correlation (above ~0.7) between two features means they're carrying largely the same information — keeping both wastes the model's capacity without adding signal.

Your hint features were mostly uncorrelated (below 0.3), which is good news: each hint feature is contributing something independent. The two most correlated pairs (F531 & F321 at 0.49, F3889 & acc_age at 0.41) are worth watching, but not problematic at these levels.

The important caveat: correlation only captures _linear_ relationships. Two features can be completely uncorrelated yet still carry redundant information in non-linear ways. Tree-based models handle this naturally — they'll simply stop using a redundant feature if another already captures the information.

---

## The General EDA Approach for Imbalanced Tabular Problems

If you were doing this from scratch on a new dataset, the mental checklist is:

1. **Check the imbalance ratio first.** This determines your entire evaluation strategy before you look at anything else. Never use accuracy.

2. **Hunt for data leakage early.** Any feature with a suspiciously strong relationship to the target — especially identifiers, timestamps, or anything encoding _when_ the label was assigned — is suspect.

3. **Use ratios and grouped statistics, not just visualisations.** On skewed data, plots lie. Numbers don't.

4. **Check both mean and median for every claim.** If they tell different stories, there are significant outliers. The median is usually more trustworthy for financial data.

5. **Treat correlations as approximate.** They flag obvious redundancy but don't guarantee independence. Let the model's feature importances (Phase 3) make the final call.

6. **Encode with intent.** Ask whether order exists before choosing an encoding strategy. It takes thirty seconds and prevents subtle errors.

7. **Low signal from individual features is normal.** In fraud detection, no single feature is decisive. Signal emerges from _combinations_ of features. That's what Phase 3 is for.

---

## Numbers to Carry Forward

| Feature                 | Signal Type                                    | Note                                |
| ----------------------- | ---------------------------------------------- | ----------------------------------- |
| F2678                   | Strong — mules 97% lower                       | Most reliable signal                |
| F2122                   | Strong — mules 9x lower                        | Very clean difference               |
| F2956                   | Moderate — mules ~50% lower                    | Reliable                            |
| F2737                   | Moderate — mules ~55% lower                    | Reliable                            |
| F1692                   | Moderate — mules ~57% lower                    | Reliable                            |
| F670                    | Moderate — mules 2.6x higher                   | Only feature where mules are higher |
| F3836                   | Weak — mean misleading, medians near identical | Don't prioritise                    |
| F527, F531, F321, F3894 | Weak — ratio within 5% of 1.0                  | Useless individually                |

**Correlated pairs to potentially combine in Phase 3:**

- F531 & F321 (0.49)
- F3889 & acc_age (0.41)
- F3894 & acc_age (0.40)
