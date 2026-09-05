# Phase 5 Cross-Validation Results

## Out-of-Fold (OOF) Results

```text
              precision    recall  f1-score   support

  Legitimate       1.00      1.00      1.00      7200
        Mule       0.77      0.77      0.77        65

    accuracy                           1.00      7265
   macro avg       0.88      0.88      0.88      7265
weighted avg       1.00      1.00      1.00      7265
```

**OOF ROC-AUC:** 0.9649

## Confusion Matrix (threshold=0.450)

| | Predicted Legit | Predicted Mule | |
|---|---|---|---|
| **Actual Legit** | 1797 | 4 | *(False Alarms)* |
| **Actual Mule** | 3 | 13 | *(Caught)* |

## Operational Metrics

- **Mules caught:** 13 / 16 (81.2% recall)
- **False alarms:** 4 / 1801 (0.22% of legit accounts flagged)
- **Investigation load:** 1 real mule per 1.3 alerts
