import pandas as pd
mydf = pd.read_csv("iith boihackathon problem 2 data.csv")
groupable = ["F3886", "F3887", "F3889", "F3890", "F3891", "F3892", "F3893"]
obj_cols = ['F2230', 'F3886', 'F3888', 'F3889', 'F3890', 'F3891', 'F3892', 'F3893']
hints = ["F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122", "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887", "F3889", "F3891", "F3894",]
mule_indicator = ["F3924"]

print(mydf.shape)
mydf.info()

print("\n Displaying null counts: \n")
null_counts = mydf[hints].isnull().sum()
print(null_counts[null_counts > 0])

print("\n Displaying number of mule accounts: \n")
print(mydf["F3924"].value_counts())
print(mydf["F3924"].value_counts(normalize=True))

# print(mydf[hints].describe().to_string())
# hint_nulls = mydf[hints].isnull().sum()
# print(hint_nulls[hint_nulls > 0])

print("\n Null percentages of hints: \n")
hints_only = mydf[hints]
null_percent = (hints_only.isnull().sum()/len(mydf)) *100
print(null_percent[null_percent > 0].sort_values(ascending=False))

print("\n Object types information: \n")
print(mydf.select_dtypes(include='object').columns.tolist())
print(mydf.select_dtypes(include='object').head())

print("\n Printing unique objects: \n")
for col in obj_cols:
    print(f"\n{col}: {mydf[col].nunique()} unique values")
    print(mydf[col].value_counts().head())