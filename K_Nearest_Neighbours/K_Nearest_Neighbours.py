import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle

df = pd.read_csv('./Dataset/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)  # Replacing missing values with outliers 
df.drop(['id'], 1, inplace=True)  # 1 is for columns

X = np.array(df.drop(['class'], 1))  # features
y = np.array(df['class']) # Lables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#   clf = neighbors.KNeighborsClassifier(n_jobs=-1)
#   clf.fit(X_train, y_train)
#   with open('KNearestNeighbors.pickle', 'wb') as f:
#		clf = pickle.dump(clf, f)

pickle_in = open('KNearestNeighbors.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,6,4,2,1,5,6,2]])

prediction = clf.predict(example_measures)
print(prediction)