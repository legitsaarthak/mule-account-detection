# Phase 4 Results

**X shape:** (9082, 50)  
**Mules:** 81 | **Legit:** 9001 | **Positive rate:** 0.89%  
**Nulls remaining:** 0

## Logistic Regression

```text
              precision    recall  f1-score   support

       Legit     0.9993    0.8434    0.9148      1801
        Mule     0.0505    0.9375    0.0958        16

    accuracy                         0.8442      1817
   macro avg     0.5249    0.8905    0.5053      1817
weighted avg     0.9910    0.8442    0.9076      1817
```

**ROC-AUC:** 0.9672  
*Confusion matrix saved: `cm_logistic_regression.png`*

## Random Forest

```text
              precision    recall  f1-score   support

       Legit     0.9961    0.9994    0.9978      1801
        Mule     0.9000    0.5625    0.6923        16

    accuracy                         0.9956      1817
   macro avg     0.9481    0.7810    0.8450      1817
weighted avg     0.9953    0.9956    0.9951      1817
```

**ROC-AUC:** 0.9918  
*Confusion matrix saved: `cm_random_forest.png`*

---

## Cross-Validation Results (5-Fold Stratified)

### Logistic Regression
- **Recall:** 0.8515 ± 0.0644
- **Precision:** 0.0552 ± 0.0058
- **F1-score:** 0.1036 ± 0.0106
- **ROC-AUC:** 0.9330 ± 0.0207

### Random Forest
- **Recall:** 0.5037 ± 0.1577
- **Precision:** 0.8363 ± 0.1505
- **F1-score:** 0.6230 ± 0.1592
- **ROC-AUC:** 0.9735 ± 0.0201

*Saved: `roc_curve_baselines.png`*

## Summary

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.0505 | 0.9375 | 0.0958 | 0.9672 |
| Random Forest | 0.9000 | 0.5625 | 0.6923 | 0.9918 |

> **Phase 4 complete. Baselines established.**  
> Target for Phase 5 (XGBoost + SMOTE): beat these numbers, especially Recall.
