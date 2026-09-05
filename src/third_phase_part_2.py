import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# X_final = pd.read_csv('final_dataset_with_target.csv')
# y = X_final['target']  
# X_final.drop(columns= ['target'])

X_final = pd.read_csv('final_dataset_with_target.csv')
y = X_final['target']                                    # extract label first
X_final = X_final.drop(columns=['target', 'IndexSS'])    # THEN drop both

# Quick sanity check before training
assert 'target' not in X_final.columns
assert 'IndexSS' not in X_final.columns
print(f"Shape: {X_final.shape}")  # should be (9082, 2031)
print(f"Mules: {y.sum()} | Legit: {(y==0).sum()}")
# then your RF code as before

# TRAINING RANDOM FOREST ALGORITHM TO GET ROUGHLY THE MOST IMPORTANT FEATURES
rf = RandomForestClassifier(
    n_estimators=500,          # more trees, more stable (takes ~30-60s on your CPU)
    max_depth=None,            # let trees grow fully
    min_samples_leaf=3,        # but each leaf must have ≥3 samples (prevents memorizing individual mules)
    min_samples_split=6,       # need ≥6 samples to attempt a split
    max_features='sqrt',       # each tree sees √(n_features) random features per split (default, but be explicit)
    class_weight='balanced',   # keep this
    random_state=42,
    n_jobs=-1
)
rf.fit(X_final, y)
print('Done.')

# EXTRACTING AND RANKING IMPORTANT FEATURES
importances = pd.Series(rf.feature_importances_, index = X_final.columns)
importances = importances.sort_values(ascending=False)
print("\nTop 30 features:")
print(importances.head(30))

# PLOTTING TOP 30 FEATURES
plt.figure(figsize=(10, 8))
importances.head(30).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 30 feature importances (RandomForest)')
plt.xlabel('Importance score')
plt.tight_layout()
plt.show()

# SELECTING TOP 50 FEATURES
top_features = importances.head(50).index.to_list()

# WHICH OF MY FEATURES SHOWED UP IN TOP 50 ?
new_cols = [
    'F531/F321', 'F3889/acc_age', 'F3894/acc_age', # Fixed names
    'F2678*F2122', 'F2678*F1692', 'F2122*F1692',
    'F670/F2678', 'F670/F2122', 'sum_of_lows', 
    'F2678_zscore', 'F2122_zscore', 'F670_zscore',
    'F2956_zscore', 'F2737_zscore'                 # Removed F1692_zscore
]
engineered = [c for c in new_cols if c in top_features]
print(f"\nEngineered features in top 50: {engineered}")

# FINAL DATASET WITH TOP 50 FEATURES
X_selected = X_final[top_features].copy()
print(f"\nFinal feature set shape: {X_selected.shape}")

# # Do these features look like normal account metrics or metadata ?
# print(X_final[['F3898', 'F3806', 'F3801']].describe())
# # Are they suspiciously clean/round numbers ?
# # Are they binary (0/1 only) ?
# mydf = pd.read_csv('final_dataset_with_target.csv')
# print(mydf.groupby('target')['F3898'].value_counts())
# print(mydf.groupby('target')['F3806'].value_counts())
# print(mydf.groupby('target')['F3801'].value_counts())
# FEATURES ARE CLEAN, NO DATA LEAKAGE

X_selected.to_csv('X_model_ready.csv', index=False)
y.to_csv('y.csv', index=False)
print("Saved. Phase 3 done.")