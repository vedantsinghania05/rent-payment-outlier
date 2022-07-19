import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools


df = pd.read_csv('/Users/vedantsinghania/code/labeled.csv')

full_cols =  {0: 'GLCODE', 1: 'GLNAME', 2: 'PROPERTY', 
3: 'PROPERTYNAME', 4: 'UNIT', 5: 'DESCRIPTION', 6: 'REFERENCE', 
7: 'DEBITCREDIT', 8: 'REMARKS', 9: 'DATEDAY', 10: 'DATEMO',
11: 'PERIODDAY', 12: 'PERIODMO', 13: 'AMOUNT'}

columns = [
    [0,1,2,3,4,5,6,7,8,10,11,12,13] #CONTROL
]

'''
Notes

'''
parameters = [
    [1] #n_neighbors
]

'''
CONTROL
n_neighbors = 1 (for now)
'''

parameter_list = list(itertools.product(*parameters))

total_exp = len(parameter_list) * len(columns)
done_exp = 0

report_df = pd.DataFrame()


def make_model(columns, n_neighbors):
    X = pd.DataFrame(df, columns = columns)
    y = pd.Series(data = df['OUTLIER'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report(columns, y_test, y_pred, n_neighbors)


def report(columns, y_test, y_pred, n_neighbors):

    row = {
        'AS': round(accuracy_score(y_test, y_pred)*100, 3), 
        'NN': n_neighbors, 
        'Outlier': pd.Series(y_pred).value_counts()[1],
        'Inlier': pd.Series(y_pred).value_counts()[0],
        'Columns': columns
        }

    global report_df
    report_df = report_df.append(row, ignore_index = True)


print('')

for e in columns:
    accessed_list = [full_cols.get(key) for key in e]

    for f in parameter_list:
        make_model(accessed_list, f[0])
        done_exp += 1
        print('{}/{} experiments completed'.format(done_exp, total_exp), end='\r')

print('')
print(report_df.to_string())
print('')

# precision: out of everything the model thinks its right, how many are actually right
# recall: out of all the actual, how many times you are correct