import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
mydf = pd.read_csv('iith boihackathon problem 2 data.csv')
groupable = ['F3886', 'F3887', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
# obj_cols = ['F2230', 'F3886', 'F3888', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
hints = ['F115', 'F321', 'F527', 'F531', 'F670', 'F1692', 'F2082', 'F2122', 'F2582', 'F2678', 'F2737', 'F2956', 'F3043', 'F3836', 'F3887', 'F3889', 'F3891', 'F3894',]
mule_indicator = ['F3924']
matplotlib.use('Agg')

# FINDING COLUMNS WITH MORE THAN 50% NULLS AND DROPPING THEM 
hints_only = mydf[hints]
null_percent = (mydf.isnull().sum()/len(mydf)) * 100
cols_to_drop = null_percent[null_percent > 50].index
mydf.drop(columns=cols_to_drop, inplace=True)
# print(f'Dropped {len(cols_to_drop)} columns. Remaining: {mydf.shape[1]}')

# DROPPING TARGET VARIABLE AND F2230 BECAUSE OF THE MASSIVE PROBLEM DISCOVERED BELOW
X = mydf.drop(columns=['F2230','F3924'])
Y = mydf['F3924']
selector = VarianceThreshold(threshold=0.01)

# SELECTING NUMERIC ONLY COLUMNS FOR FINDING VARIANCE
numeric_cols = X.select_dtypes(exclude='object').columns
X_numeric = X[numeric_cols]

# FINDING COLUMNS WITH VARIANCE LESS THAN 1% AND DROPPING THEM
selector.fit(X_numeric)
low_var_cols = numeric_cols[~selector.get_support()]
X.drop(columns=low_var_cols, inplace=True)
# print(f'Dropped {len(low_var_cols)} near-zero variance columns')
# print(X.shape)

X['acc_age'] = (
    pd.Timestamp.today() - pd.to_datetime(X['F3888'])
).dt.days

# CONVERTING DATE INTO ACCOUNT AGE IN DAYS
X.drop(columns = ['F3888'], inplace=True)
# print(X['acc_age'])
# print(X['F3888'])

obj_cols = X.select_dtypes(include='object').columns
# print(f'Object columns remaining: {obj_cols.tolist()}')
# for col in obj_cols:
#     print(f'\n{col}:')
#     print(X[col].value_counts().to_string())

# MASSIVE PROBLEM: ALL LEGITIMATE ACCOUNTS ARE IN Oct25, ALL MULES AREN'T
# print(mydf.groupby('F3924')['F2230'].value_counts())

# ESTABLISHING ORDER FOR F3889: THEY LOOK LIKE AGE BUCKETS, THEREFORE ENCODING IT ORDINALLY
age_order = {'L7D':1,'L14D':2,'L31D':3,'L90D':4,'L180D':5,'L365D':6,'G365D':7}
X['F3889'] = X['F3889'].map(age_order)

# ENCODING THE REMANING OBJECT TYPES INTO LABELS (better than OHE for tree based models like XGBoost)
le = LabelEncoder()
for col in ['F3886', 'F3890', 'F3891', 'F3892', 'F3893']:
    X[col] = le.fit_transform(X[col].astype(str))

# FILLING REMAINING NUMERIC NULLS WITH MEDIANS
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# CREATING A COPY OF FEATURE MATRIX IN A WAY THAT MATPLOTLIB CAN REFER TO IT
plot_df = X_imputed.copy()
plot_df['target'] = Y.values

# CREATING A CLEAN HINTS ARRAY, MAKING SURE ALL HINTS ARE PRESENT AND HAVE THEIR UPDATED NAMES
hints_clean = [f for f in hints if f in plot_df.columns]
hints_clean = [f if f != 'F3888' else 'acc_age' for f in hints_clean]
hints_clean.append('acc_age')
# USING MATPLOTLIB TO PLOT HISTOGRAMS OF HINTS, LOOK UP CODE TO GET MORE EXPLANATION
fig, axes = plt.subplots(6, 3, figsize=(18,20))
axes = axes.flatten()
for i,hint in enumerate(hints_clean):
    ax = axes[i]
    plot_df[plot_df['target'] == 0][hint].hist(ax=ax, bins=50, alpha=0.6, label='Legitimate', color='steelblue')
    plot_df[plot_df['target'] == 1][hint].hist(ax=ax, bins=50, alpha=0.6, label='Mule', color='crimson')
    ax.set_title(hint)
    ax.legend(fontsize=7)
# (hiding empty spaces since hints are less than 18)
for j in range(len(hints_clean), len(axes)):
    axes[j].set_visible(False)
# (finally printing and saving the distributions)
plt.tight_layout()
plt.savefig('hint_distributions.png', dpi=150)
print("Saved: hint_distributions.png")

hints_available = [f for f in hints_clean if f in plot_df.columns]
# hints_available.append('acc_age')
# PRINTING AND SAVING CORRELATION HEATMAP FOR ALL HINT FEATURES
plt.figure(figsize=(14,10))
sns.heatmap(
    plot_df[hints_available].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True
)
plt.title('Correlation Between Hint Features')
plt.tight_layout()
plt.savefig('hint_correlation.png', dpi=150)
print("Saved: hint_correlation.png")

# CREATING STATISTICAL SUMMARY TABLE, AND MEAN RATIO OF ALL FEATURES OF LEGIT TO MULE ACCOUNTS
summary = plot_df.groupby('target')[hints_available].mean().T
summary.columns = ['Legit(mean)', 'Mule(mean)']
summary['Ratio'] = summary["Mule(mean)"] / (summary['Legit(mean)'] + 1e-9)
summary = summary.sort_values('Ratio', ascending=False)
print(summary.to_string())

# LEMME CHECK SOME SHIT
# no_reds = ['F527', 'F2678', 'F2737', 'F2956', 'F3836', 'F3887', 'F3894']

# for feature in hints_available:
#     fig, axes = plt.subplots(1, 2, figsize=(14, 4))

#     plot_df[plot_df['target'] == 0][feature].hist(ax=axes[0], bins=50, color='steelblue')
#     axes[0].set_title(f'{feature} — Legitimate')

#     plot_df[plot_df['target'] == 1][feature].hist(ax=axes[1], bins=50, color='crimson')
#     axes[1].set_title(f'{feature} — Mule')

#     plt.tight_layout()
#     plt.savefig(f'{feature}_zoom.png', dpi=150)
#     plt.close()
#     print(f'Saved: {feature}_zoom.png')

# Strong signal:    F670, F2678, F2122, F3836
# Moderate signal:  F2956, F2737, F1692, F2582
# Weak/noise:       F115, F321, F527, F531, F3889, F3887, F3891, F3894

# print(plot_df.groupby('target')['F3836'].describe())
print(plot_df.groupby('target')['acc_age'].describe())
print(hints_available)
# --- Save to CSV --- DROPPING F3887 BECAUSE CORRELATION TO acc_age IS 1.0
# plot_df.drop(columns=['F3887'], inplace=True)
# plot_df.to_csv('cleaned_data_with_target.csv', index=False)
# print("Saved prepared dataset to 'cleaned_data_with_target.csv'")
