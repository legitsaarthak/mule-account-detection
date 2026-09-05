# XGBoost vs. XGBoost + SMOTE Results

## XGBoost (scale_pos_weight, threshold=0.5)

```text
              precision    recall  f1-score   support

  Legitimate       1.00      1.00      1.00      1801
        Mule       0.71      0.75      0.73        16

    accuracy                           1.00      1817
   macro avg       0.85      0.87      0.86      1817
weighted avg       1.00      1.00      1.00      1817
```

**ROC-AUC:** 0.9967

## XGBoost + SMOTE (threshold=0.5)

```text
              precision    recall  f1-score   support

  Legitimate       1.00      1.00      1.00      1801
        Mule       0.80      0.75      0.77        16

    accuracy                           1.00      1817
   macro avg       0.90      0.87      0.89      1817
weighted avg       1.00      1.00      1.00      1817
```

**ROC-AUC:** 0.9954

## ROC-AUC Comparison

- **XGB scale_pos_weight:** 0.9967
- **XGB + SMOTE:** 0.9954
