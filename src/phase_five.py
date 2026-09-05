import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score
)

# READING DATA
X = pd.read_csv('X_model_ready.csv')
y = pd.read_csv('y.csv').squeeze()
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

# ESTABLISHING scale_pos_weight 
# neg_count = (y_train==0).sum()
# pos_count = (y_train==1).sum()
# spw = neg_count/pos_count

# # XGBOOST PIPELINE: NO SCALING NEEDED BECAUSE TREE BASED MODEL
# xgb_base = ImbPipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('xgb', XGBClassifier(
#         n_estimators = 300,
#         max_depth = 4,
#         min_child_weight = 3,
#         learning_rate = 0.05,
#         scale_pos_weight = spw,
#         subsample = 0.8,
#         colsample_bytree = 0.8,
#         random_state = 42,
#         eval_metric = 'logloss',
#         verbosity = 0
#     ))
# ])
# xgb_base.fit(X_train, y_train)
# y_pred_xgb = xgb_base.predict(X_test)
# y_prob_xgb = xgb_base.predict_proba(X_test)[:, 1]
# # EVALUATING AT DEFAULT 0.5 THRESHOLD
# print("=== XGBoost (scale_pos_weight, threshold=0.5) ===")
# print(classification_report(y_test, y_pred_xgb, target_names=['Legitimate', 'Mule']))
# print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# ───────────────────────────────────────────────────────────────────────────────────────────────
# XGBOOST + SMOTE PIPELINE
# ───────────────────────────────────────────────────────────────────────────────────────────────
xgb_smote = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(
        sampling_strategy = 0.3,
        k_neighbors = 5,
        random_state = 42
    )),
    ('xgb', XGBClassifier(
        n_estimators = 500,
        learning_rate = 0.05,
        max_depth = 4,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42,
        eval_metric = 'aucpr',
        verbosity = 0,
        gamma = 1.0,
        min_child_weight = 3,
    ))
])
xgb_smote.fit(X_train, y_train)
y_pred_smote = xgb_smote.predict(X_test)
y_prob_smote = xgb_smote.predict_proba(X_test)[:, 1]
# EVALUATING AT DEFAULT 0.5 THRESHOLD
print("=== XGBoost + SMOTE (threshold=0.5) ===")
print(classification_report(y_test, y_pred_smote, target_names=['Legitimate', 'Mule']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_smote):.4f}")

# ───────────────
# GETTING RESULTS 
# ───────────────
# results = {
#     'XGB scale_pos_weight': roc_auc_score(y_test, y_prob_xgb),
#     'XGB + SMOTE':         roc_auc_score(y_test, y_prob_smote),
# }
# for name, auc in results.items():
#     print(f"{name}: {auc:.4f}")
# print("\n \n \n")
# ───────────────────────────────────────────────────────────────────────────────────────────────
# # GETTING THRESHOLD TABLE AND SAVING GRAPHS
# ───────────────────────────────────────────────────────────────────────────────────────────────
# threshold_results = []
# for thr in np.arange(0.05, 0.95, 0.05):
#     preds = (y_prob_smote >= thr).astype(int)
#     threshold_results.append({
#         'Threshold': round(thr, 2),
#         'Precision': precision_score(y_test, preds, zero_division=0),
#         'Recall': recall_score(y_test, preds),
#         'F1': f1_score(y_test, preds)
#     })
# threshold_df = pd.DataFrame(threshold_results)
# print("\nThreshold sweep:")
# print(threshold_df)
# best_probs = y_prob_smote
# precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precisions[:-1], label='Precision', color='steelblue')
# plt.plot(thresholds, recalls[:-1], label='Recall', color='tomato')
# plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold (0.5)')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title('Precision vs Recall at Every Threshold(XGB+SMOTE)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('precision_recall_threshold(XGB+SMOTE).png', dpi=150)
# plt.show()

# threshold_results = []
# for thr in np.arange(0.05, 0.95, 0.05):
#     preds = (y_prob_xgb >= thr).astype(int)
#     threshold_results.append({
#         'Threshold': round(thr, 2),
#         'Precision': precision_score(y_test, preds, zero_division=0),
#         'Recall': recall_score(y_test, preds),
#         'F1': f1_score(y_test, preds)
#     })
# threshold_df = pd.DataFrame(threshold_results)
# print("\nThreshold sweep:")
# print(threshold_df)
# best_probs = y_prob_xgb
# precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precisions[:-1], label='Precision', color='steelblue')
# plt.plot(thresholds, recalls[:-1], label='Recall', color='tomato')
# plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default threshold (0.5)')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title('Precision vs Recall at Every Threshold(XGB using spw)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('precision_recall_threshold(XGB using spw).png', dpi=150)
# plt.show()
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# VERDICT: WE GET THE BEST PRECISION-RECALL RATIO WITH XGB+SMOTE AT A PROBABILITY THRESHOLD OF 0.45
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
chosen_threshold = 0.45
y_pred_tuned = (y_prob_smote >= chosen_threshold).astype(int)
# ───────────────────────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION TIME: FIRST OUT-OF-FOLD CV, THEN NORMAL CV
# ───────────────────────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = cross_val_predict(
    xgb_smote,
    X_train,
    y_train,
    cv=cv,
    method = 'predict_proba'
)[:, 1]
oof_preds = (oof_probs >= chosen_threshold).astype(int)
print("=== Cross-Validation (Out-of-Fold) Results ===")
print(classification_report(y_train, oof_preds, target_names=['Legitimate', 'Mule']))
print(f"OOF ROC-AUC: {roc_auc_score(y_train, oof_probs):.4f}")

# metrics_for_cv = ['recall', 'precision', 'f1', 'roc_auc']
# for model_name, pipeline in [('XGBoost+SMOTE', xgb_smote)]:
#     scores = cross_validate(pipeline, X, y, cv=cv, scoring=metrics_for_cv, n_jobs=-1)
#     print(f"\n{model_name}:")
#     for metric in metrics_for_cv:
#         vals = scores[f'test_{metric}']
#         print(f"  {metric:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

# CONFUSION MATRIX
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