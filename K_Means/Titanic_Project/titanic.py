import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
PClass
Survival
Name
Age
Sex
Ticket
sibsp Number(# of siblings/spouses onboard)
parch Number
fare
cabin
embarked port of embarkation
body identification number (body)
'''

df = pd.read_excel('./data/titanic.xls')
df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

# K Means Algorithim only deals with numeric data.
#
# Here we need to classify survive
# or dead. Therefore value of n-clusters
# will be 2.

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents) # {'female', 'male'}
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2) # default n_clusters = 8
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float)) #  To avoid any discrepencies we convert it into float.
    predict_me = predict_me.reshape(-1,len(predict_me)) # predict_me is one row of the data_set.
    prediction = clf.predict(predict_me) # prediction = [0] or [1]
    if prediction[0] == y[i]:
        correct+=1

#print(X)
#print(np.array(X[0].astype('float')))
print(correct/len(X))



 
















        
