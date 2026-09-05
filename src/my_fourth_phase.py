import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    precision_score, 
    recall_score, 
    f1_score,
    
)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# READING DATA
X = pd.read_csv('X_model_ready.csv')
y = pd.read_csv('y.csv').squeeze()
print(f"X shape: {X.shape}")
print(f"Mules: {y.sum()} | Legit: {(y==0).sum()} | Positive rate: {y.sum()/len(y)*100:.2f}%")
print(f"Nulls remaining: {X.isnull().sum().sum()}")

# FORCE INCLUDING sum_of_lows SINCE IT DIDN'T MAKE IT TO TOP 50
if 'sum_of_lows' not in X.columns:
    raw = pd.read_csv('final_dataset_with_target.csv')
    X['sum_of_lows'] = raw['sum_of_lows'].values

# SPLITTING THE MODEL READY DATASET INTO TRAINING AND TESTING DATA WITH EQUAL PERCENTAGES OF MULES (STRATIFICATION)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

# EVALUATION FUNCTION: PRINTS ALL THE IMPORTANT INFO ABOUT A GIVEN TRAINED MODEL
def evaluate(model_name, pipeline, X_test, y_test, save_cm=True):
    
    # PRINTING PRECISION, RECALL, F1-SCORE, SUPPORT
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Mule'], digits=4))

    # PRINTING ROC-AUC SCORE
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    # SAVING CONFUSION MATRIX TO AN IMAGE
    if save_cm:
        fig, ax = plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=['Legit', 'Mule'],
            cmap='Blues',
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        fname = f"cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Confusion matrix saved: {fname}")
    
    return y_prob

# ───────────────────────────────────────────────────────────────────────────────────────────────
# FIRST MODEL: LOGISTIC REGRESSION USING PIPELINE
# setting a baseline. this is the simplest model. any model that can't beat this is useless.
# ───────────────────────────────────────────────────────────────────────────────────────────────
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        C=1.0
    ))
])
lr_pipeline.fit(X_train, y_train)
lr_probs = evaluate("Logistic Regression", lr_pipeline, X_test, y_test)

# ───────────────────────────────────────────────────────────────────────────────────────────────
# SECOND MODEL: RANDOM FOREST (AGAIN)
# setting baseline for XGBoost. 
# ───────────────────────────────────────────────────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ('clf', RandomForestClassifier(
        n_estimators = 700,
        max_depth = None,
        min_samples_leaf = 3,
        min_samples_split = 6,
        max_features = 'sqrt',
        class_weight = 'balanced_subsample',
        oob_score = True,
        random_state = 42,
        n_jobs = -1
    ))
])
rf_pipeline.fit(X_train, y_train)
rf_probs = evaluate('Random Forest', rf_pipeline, X_test, y_test)

# # TESTING MUTIPLE THRESHOLDS
# threshold_results = []

# for thr in np.arange(0.05, 0.95, 0.05):

#     preds = (rf_probs >= thr).astype(int)

#     threshold_results.append({
#         'Threshold': round(thr,2),
#         'Precision': precision_score(y_test, preds, zero_division=0),
#         'Recall': recall_score(y_test, preds),
#         'F1': f1_score(y_test, preds)
#     })

# threshold_df = pd.DataFrame(threshold_results)

# print("\nThreshold sweep:")
# print(threshold_df)

# ───────────────────────────────────────────────────────────────────────────────────────────────
# CROSS VALIDATION ON BOTH MODELS: 
# SPLITS THE DATA 5 DIFFERENT WAYS ALL WITH SAME IMBALANCE RATIO AND TRAINS AND EVALUATES 5 TIMES
# ───────────────────────────────────────────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
metrics_for_cv = ['recall', 'precision', 'f1', 'roc_auc']

print("\n" + "="*50)
print("  Cross-Validation Results (5-Fold Stratified)")
print("="*50)

for model_name, pipeline in [('Random Forest', rf_pipeline)]:
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=metrics_for_cv, n_jobs=-1)
    print(f"\n{model_name}:")
    for metric in metrics_for_cv:
        vals = scores[f'test_{metric}']
        print(f"  {metric:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

# ──────────────────────────────────────────────────────────────────────────────────────────
# ROC CURVE: SHOWS PRECISION AND RECALL AT EVERY POSSIBLE THRESHOLD OF PROBABILITY
# ──────────────────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for model_name, probs in [("Logistic Regression", lr_probs), ("Random Forest", rf_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

ax.plot([0,1], [0,1], 'k--', label='Random baseline (AUC = 0.500)')
ax.set_xlabel('False Positive Rate (Legit accounts flagged as mule)')
ax.set_ylabel('True Positive Rate (Recall — mules actually caught)')
ax.set_title('ROC Curve — Baseline Models')
ax.legend()
plt.tight_layout()
plt.savefig('roc_curve_baselines.png', dpi=150)
print("\nSaved: roc_curve_baselines.png")

# ─────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────
rows = []
for model_name, pipeline, probs in [
    
    ("Random Forest",       rf_pipeline, rf_probs),
]:
    y_pred = pipeline.predict(X_test)
    rows.append({
        'Model':     model_name,
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1':        round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, probs), 4),
    })

results = pd.DataFrame(rows)
print("\n")
print(results.to_string(index=False))
print("\nPhase 4 complete. Baselines established.")
print("Target for Phase 5 (XGBoost + SMOTE): beat these numbers, especially Recall.")

# RESAVE X_model_ready.csv WITH sum_of_lows INCLUDED
X.to_csv('X_model_ready.csv', index=False)
print("Resaved X_model_ready.csv with sum_of_lows included.")



# print("\n \n \n \n \n")
# # RF out-of-fold at threshold 0.25
# rf_oof_probs = cross_val_predict(
#     rf_pipeline,
#     X_train, y_train,
#     cv=cv,
#     method='predict_proba'
# )[:, 1]

# rf_oof_preds = (rf_oof_probs >= 0.25).astype(int)
# print(classification_report(y_train, rf_oof_preds, target_names=['Legitimate', 'Mule']))
# print(f"RF OOF ROC-AUC: {roc_auc_score(y_train, rf_oof_probs):.4f}")
