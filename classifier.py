import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/vedantsinghania/code/labeled.csv')
df.head()

X = df.copy()
y = df['OUTLIER'].copy()

X = X.drop(['OUTLIER'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Score: {}%".format(round(accuracy_score(y_test, y_pred)*100, 3)))
print(pd.Series(y_pred).value_counts())