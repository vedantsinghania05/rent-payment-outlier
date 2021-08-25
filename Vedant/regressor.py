import pandas as pd
import numpy as np
import math
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/vedantsinghania/code/export_yardi_jourentline.csv', lineterminator='\n')
df.columns = df.columns.map(lambda a: a.strip())

df['DATE'] = df['DATE'].map(lambda b: datetime.strptime(b, ' %m/%d/%Y'))
df['PERIOD'] = df['PERIOD'].map(lambda b: datetime.strptime(b, ' %m/%d/%Y'))

df['DATEDAY'] = df['DATE'].map(lambda c: int(c.day))
df['DATEMO'] = df['DATE'].map(lambda c: int(c.month))

df['PERIODDAY'] = df['PERIOD'].map(lambda d: int(d.day))
df['PERIODMO'] = df['PERIOD'].map(lambda d: int(d.month))

df['AMOUNT'] = df['AMOUNT'].map(lambda a: math.ceil(a))

X = df.copy()
X = X.drop(['AMOUNT'], axis=1)

y = df['AMOUNT'].copy()

X = X.drop(columns=['CONTROL', 'BALANCE', 'DATE', 'PERIOD', 'BUILDING', 'SPECIALCIRCUMSTANCE', 'USAGEAMOUNT', 'USAGETYPE'])

cat_cols = [col for col in X.columns if X[col].dtype == 'object']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

label_X_train = X_train.copy()
label_X_test = X_test.copy()
le = LabelEncoder()

for col in cat_cols:
    le.fit(label_X_train[col])

    label_X_test[col] = label_X_test[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')

    label_X_train[col] = le.transform(label_X_train[col])
    label_X_test[col] = le.transform(label_X_test[col])

X_train = label_X_train.copy()
X_test = label_X_test.copy()

model = XGBRegressor(random_state=1, n_jobs=-1, learning_rate=0.22, n_estimators=10000, max_depth=7, objective='count:poisson', colsample_bytree=0.2)
model.fit(X_train, y_train, early_stopping_rounds=25, eval_set=[(X_test, y_test)], verbose=False)

training_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# R² (training): 0.9821118335775684
# R² (test): 0.6619346763391929

X_train['P_AMOUNT'] = training_preds.copy()
X_test['P_AMOUNT'] = test_preds.copy()
X_train['AMOUNT'] = y_train.copy()
X_test['AMOUNT'] = y_test.copy()

df_for_csv = pd.concat([X_test, X_train], ignore_index = True)

df_for_csv['OUTLIER'] = [1 if abs( 100 - ((df_for_csv['P_AMOUNT'][i]/df_for_csv['AMOUNT'][i])*100) ) > 50 else 0 for i in range(0, len(df_for_csv))]
df_for_csv = df_for_csv.drop(['P_AMOUNT'], axis=1)

print(df_for_csv.head())
print(df_for_csv.tail())
print(df_for_csv['OUTLIER'].value_counts())

df_for_csv.to_csv(path_or_buf = '../../labeled.csv', index=False)