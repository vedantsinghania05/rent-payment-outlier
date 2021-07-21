import pandas as pd
import numpy as np
import math
from datetime import datetime
from xgboost import XGBRegressor
import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/vedantsinghania/code/export_yardi_jourentline.csv', lineterminator='\n')
df.columns = df.columns.map(lambda a: a.strip())

df['DATE'] = df['DATE'].map(lambda b: datetime.strptime(b, ' %m/%d/%Y'))
df['PERIOD'] = df['PERIOD'].map(lambda b: datetime.strptime(b, ' %m/%d/%Y'))

df['DATEDAY'] = df['DATE'].map(lambda c: int(c.day))
df['DATEMO'] = df['DATE'].map(lambda c: int(c.month))
df['DATEYR'] = df['DATE'].map(lambda c: int(c.year))

df['PERIODDAY'] = df['PERIOD'].map(lambda d: int(d.day))
df['PERIODMO'] = df['PERIOD'].map(lambda d: int(d.month))
df['PERIODYR'] = df['DATE'].map(lambda d: int(d.year))

df['AMOUNT'] = df['AMOUNT'].map(lambda a: math.ceil(a))

columns = [
    ['GLCODE', 'GLNAME', 'PROPERTY', 'PROPERTYNAME', 'UNIT', 'DESCRIPTION', 'REFERENCE', 'DEBITCREDIT', 'REMARKS', 'DATEDAY', 'DATEMO', 'PERIODDAY', 'PERIODMO']
]

parameters = [
    [10000],
    [0.22],
    [7],
    [0.2],
    ['count:poisson']
]

#parameters = [n_estimators, learning_rate, max_depth, colsample_bytree, objective]
parameter_list = list(itertools.product(*parameters))


def make_model(columns, n_estimators, learning_rate, max_depth, colsample_bytree, objective):
    X = pd.DataFrame(df, columns = columns)
    y = pd.Series(data = df['AMOUNT'])
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

    model = XGBRegressor(random_state = 1, n_jobs = -1, learning_rate = learning_rate, n_estimators = n_estimators, max_depth = max_depth, objective = objective, colsample_bytree = colsample_bytree)
    model.fit(X_train, y_train, early_stopping_rounds = 25, eval_set = [(X_test, y_test)], verbose=False)
    training_preds = model.predict(X_train)
    test_preds = pd.Series(model.predict(X_test))
    report(columns, y_train, y_test, training_preds, test_preds, n_estimators, learning_rate, max_depth, colsample_bytree, objective)


def report(columns, y_train, y_test, training_preds, test_preds, n_estimators, learning_rate, max_depth, colsample_bytree, objective):
    percents_list = []

    for i in range(0, len(y_test)):
        real = y_test[i]
        pred = test_preds[i]
        percent_of_real = (pred/real)*100
        percent_off = abs(100-percent_of_real)
        percents_list.append(round(percent_off, 2))

    percents_df = pd.DataFrame(percents_list, columns=['percent_off'])
    percents_df.head()

    percents_df['Category (%)'] = percents_df.apply(lambda p: categorize_percents(p['percent_off']), axis=1)
    percent_range_df = percents_df.groupby('Category (%)').size().reset_index(name='Count')
    percent_range_df['Percent of Total (%)'] = percent_range_df['Count']/len(percents_df.index)*100

    print('Parameters:', 'n_estimators =', n_estimators, 'learning_rate =', learning_rate, 'max_depth =', max_depth, 'colsample_bytree =', colsample_bytree, 'objective =', objective)
    print('')
    print('Columns:', columns)
    print('')
    print('Scores')
    print('R\u00b2 (training):', r2_score(y_train, training_preds))
    print('R\u00b2 (test):', r2_score(y_test, test_preds))
    print('MAE (test):', mean_absolute_error(y_test, test_preds))
    print('RMSE (test):', mean_squared_error(y_test, test_preds, squared=False))
    print('')
    print('% Off Distribution')
    print(percent_range_df)
    print('')
    print('Preds')
    print('Avg:', test_preds.mean())
    print('Min:', test_preds.min())
    print('Max:', test_preds.max())
    print('')
    print('Actuals')
    print('Avg:', y_test.mean())
    print('Min:', y_test.min())
    print('Max:', y_test.max())
    print('')


def categorize_percents(p):
    if p >= 0 and p < 5:
        return 5
    elif p >= 5 and p < 10:
        return 10
    elif p >= 10 and p < 20:
        return 20
    elif p >= 20 and p < 50:
        return 50
    elif p >= 50 and p < 100:
        return 100
    elif p >= 100 and p < 200:
        return 200
    elif p >= 200 and p < 500:
        return 500
    elif p >= 500 and p < 1000:
        return 1000
    elif p >= 1000:
        return 10000
    else:
        return 'unknown'


for e in columns:
    columns = e
    for f in parameter_list:
        make_model(columns, f[0], f[1], f[2], f[3], f[4])



