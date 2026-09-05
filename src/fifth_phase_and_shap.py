import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import shap
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

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# VERDICT: WE GET THE BEST PRECISION-RECALL RATIO WITH XGB+SMOTE AT A PROBABILITY THRESHOLD OF 0.45
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
chosen_threshold = 0.45
y_pred_tuned = (y_prob_smote >= chosen_threshold).astype(int)

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION: OUT-OF-FOLD
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = cross_val_predict(
    xgb_smote,
    X_train,
    y_train,
    cv=cv,
    method = 'predict_proba'
)[:, 1]
oof_preds = (oof_probs >= chosen_threshold).astype(int)
print("=== Cross-Validation (Out-of-Fold)(using threshold 0.45) Results ===")
print(classification_report(y_train, oof_preds, target_names=['Legitimate', 'Mule']))
print(f"OOF ROC-AUC: {roc_auc_score(y_train, oof_probs):.4f}")

# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# SHAP: SHapley Additive exPlanations
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# xgb_model = xgb_smote.named_steps['xgb']
# imputer = xgb_smote.named_steps['imputer']

# X_test_imputed = imputer.transform(X_test)
# X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test_imputed)

# ───────────────────────────────────────────────────
# SHAP: PLOTTING BEESWARM AND MEAN ABSOLUTE IMPACT
# ───────────────────────────────────────────────────
# shap.summary_plot(shap_values, X_test_imputed, show=False)
# plt.title("SHAP Summary — Feature Impact on Mule Prediction")
# plt.tight_layout()
# plt.savefig("shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
# plt.show()

# shap.summary_plot(shap_values, X_test_imputed, plot_type='bar', show=False)
# plt.title("SHAP Feature Importance — Mean Absolute Impact")
# plt.tight_layout()
# plt.savefig("shap_bar.png", dpi=150, bbox_inches='tight')
# plt.show()

# ────────────────────────────────────────────
# SHAP: PLOTTING WATERFALLS FOR CAUGHT MULES
# ────────────────────────────────────────────
# caught_mule_indices = [
#     i for i, (true, pred) in enumerate(zip(y_test, y_pred_tuned))
#     if true == 1 and pred == 1
# ]

# for idx in caught_mule_indices:
#     shap.plots.waterfall(
#         shap.Explanation(
#             values = shap_values[idx],
#             base_values = explainer.expected_value,
#             data = X_test_imputed.iloc[idx],
#             feature_names = X_test_imputed.columns.to_list()
#         ),
#         show=False
#     )
#     plt.title(f"Why Account #{idx} Was Flagged as a Mule")
#     plt.tight_layout()
#     plt.savefig(f"shap_waterfall_mule_number_{idx}.png", dpi=150, bbox_inches='tight')
#     plt.show()

# ────────────────────────────────────────────
# SHAP: PLOTTING WATERFALL FOR ONE MISSED MULE
# ────────────────────────────────────────────
# missed_mule_indices = [
#     i for i, (true, pred) in enumerate(zip(y_test, y_pred_tuned))
#     if true == 1 and pred == 0
# ]
# if missed_mule_indices:
#     idx_missed = missed_mule_indices[0]
#     shap.plots.waterfall(
#         shap.Explanation(
#             values        = shap_values[idx_missed],
#             base_values   = explainer.expected_value,
#             data          = X_test_imputed.iloc[idx_missed],
#             feature_names = X_test_imputed.columns.tolist()
#         ),
#         show=False
#     )
#     plt.title(f"Why Account #{idx_missed} Was Missed")
#     plt.tight_layout()
#     plt.savefig("shap_waterfall_missed.png", dpi=150, bbox_inches='tight')
#     plt.show()
