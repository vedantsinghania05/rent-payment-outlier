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

full_cols =  {0: 'GLCODE', 1: 'GLNAME', 2: 'PROPERTY', 
3: 'PROPERTYNAME', 4: 'UNIT', 5: 'DATE', 6: 'PERIOD', 
7: 'DESCRIPTION', 8: 'CONTROL', 9: 'REFERENCE', 10: 'DEBITCREDIT',
11: 'BALANCE', 12: 'REMARKS', 13: 'DATEDAY', 
14: 'DATEMO', 15: 'DATEYR', 16: 'PERIODDAY', 
17: 'PERIODMO', 18: 'PERIODYR'}

columns = [
    [0,1,2,3,4,7,9,10,12,13,14,16,17], # CONTROL
    [0,1,2,3,4,7,9,10,12,13,14,16],
    [0,1,2,3,4,7,9,10,12,13,14,17],
    [0,1,2,3,4,7,9,10,12,13,16,17],
    [0,1,2,3,4,7,9,10,12,14,16,17],
    [0,1,2,3,4,7,9,10,13,14,16,17],
    [0,1,2,3,4,7,9,12,13,14,16,17],
    [0,1,2,3,4,7,10,12,13,14,16,17],
    [0,1,2,3,4,9,10,12,13,14,16,17],
    [0,1,2,3,7,9,10,12,13,14,16,17],
    [0,1,2,4,7,9,10,12,13,14,16,17],
    [0,1,3,4,7,9,10,12,13,14,16,17],
    [0,2,3,4,7,9,10,12,13,14,16,17],
    [1,2,3,4,7,9,10,12,13,14,16,17],



]

'''
Notes
Keep 0, 1, 7, 9, 10
'''
parameters = [
    [0.22], #learning_rate
    [7], #max_depth
    [0.2], #colsample_bytree
    ['count:poisson'] #objective
]

'''
CONTROL
learning_rate = 0.22
max_depth = 7
colsample_bytree = 0.2
objective = 'count:poisson'
'''

parameter_list = list(itertools.product(*parameters))

total_exp = len(parameter_list) * len(columns)
done_exp = 0

report_df = pd.DataFrame(columns=[
    'R2 (TR)', 'R2 (T)', 'MAE (T)', 'RMSE (T)', '5%', '10%', '20%', '100%', '200%', '500%', '1000%', '10000%', 'lr', 'md', 'cb', 'obj', 'columns'
    ])


def make_model(columns, learning_rate, max_depth, colsample_bytree, objective):
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

    model = XGBRegressor(random_state = 1, n_jobs = -1, learning_rate = learning_rate, n_estimators = 10000, max_depth = max_depth, objective = objective, colsample_bytree = colsample_bytree)
    model.fit(X_train, y_train, early_stopping_rounds = 25, eval_set = [(X_test, y_test)], verbose=False)
    training_preds = model.predict(X_train)
    test_preds = pd.Series(model.predict(X_test))
    report(columns, y_train, y_test, training_preds, test_preds, learning_rate, max_depth, colsample_bytree, objective)


def report(columns, y_train, y_test, training_preds, test_preds, learning_rate, max_depth, colsample_bytree, objective):
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
    percents_df = (percents_df.groupby('Category (%)').size()/len(percents_df.index)*100).reset_index(name='perc. of tot')
    percent_list = []

    for i in percents_df['perc. of tot']:
        percent_list.append(i)

    new_row = {
        'R2 (TR)': round(r2_score(y_train, training_preds), 3), 
        'R2 (T)': round(r2_score(y_test, test_preds), 3), 
        'MAE (T)': round(mean_absolute_error(y_test, test_preds), 3), 
        'RMSE (T)': round(mean_squared_error(y_test, test_preds, squared=False), 3), 
        '5%': round(percent_list[0], 3),
        '10%': round(percent_list[1], 3), 
        '20%': round(percent_list[2], 3), 
        '100%': round(percent_list[3], 3), 
        '200%': round(percent_list[4], 3),
        '500%': round(percent_list[5], 3), 
        '1000%': round(percent_list[6], 3), 
        '10000%': round(percent_list[7], 3), 
        'lr': learning_rate, 
        'md': max_depth, 
        'cb': colsample_bytree, 
        'obj': objective, 
        'columns': columns
        }

    global report_df
    report_df = report_df.append(new_row, ignore_index = True)


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


print('')

for e in columns:
    accessed_list = [full_cols.get(key) for key in e]

    for f in parameter_list:
        make_model(accessed_list, f[0], f[1], f[2], f[3])
        done_exp += 1
        print('{}/{} experiments completed'.format(done_exp, total_exp))

print('')
print(report_df.to_string())
print('')