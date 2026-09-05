import numpy as np
import pandas as pd
hints_3 = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2122', 'F2582', 'F2678', 'F2737', 'F2956', 'F3836', 'F3889', 'F3891', 'F3894', 'acc_age']
newdf = pd.read_csv('cleaned_data_with_target.csv')
TARGET = 'target'

X = newdf.drop(columns=[TARGET])
Y = newdf[TARGET]

# MAKING RATIOS WITH THE CORRELATING FEATURES WHICH WILL TELL US THE IMBALANCE (pretty useless imo)
X['F531/F321'] = X['F531'] / (X['F321'] + 1e-6)
X['F3889/acc_age'] = X['F3889'] / (X['acc_age'] + 1e-6)
X['F3894/acc_age'] = X['F3894'] / (X['acc_age'] + 1e-6)

# MULTIPLYING MULES' LOW FEATURES BY LOW FEATURES WHICH SHOULD BE SUPER LOW FOR MULES
X['F2678*F2122'] = X['F2678'] * X['F2122']
X['F2678*F1692'] = X['F2678'] * X['F1692']
X['F2122*F1692'] = X['F2122'] * X['F1692']

# RATIO OF HIGHER FOR MULES TO LOWER FOR MULES SHOULD BE EXTREMELY HIGH FOR MULES
X['F670/F2678'] = X['F670'] / (X['F2678'] + 1e-6)
X['F670/F2122'] = X['F670'] / (X['F2122'] + 1e-6)

# SUM OF ALL FEATURES THAT ARE MODERATELY TO HIGHLY LOW FOR MULES
X['sum_of_lows'] = X['F2678'] + X['F2122'] + X['F2956'] + X['F2737'] + X['F1692']

# FINDING Z-SCORE OF ALL LOWER FEATURES
for col in ['F2678', 'F2122', 'F670', 'F2956', 'F2737']:
    col_mean = X[col].mean()
    col_std = X[col].std()
    if col_std > 0:
        X[f"{col}_zscore"] = (X[col] - col_mean) / col_std

new_cols = [ col for col in X.columns if col not in newdf.columns]
print(f"New features created: {len(new_cols)}")
print(X[new_cols].isnull().sum())
print("\n \n")
print(X.shape)

final_dataset = X.copy()
final_dataset['target'] = Y.values
final_dataset.to_csv('final_dataset_with_target.csv', index=False)
print(final_dataset.shape)

