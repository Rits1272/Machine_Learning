import pandas as pd 
import quandl, math, datetime
import numpy as np 
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# cross validation is used to shuffle the data so that the data do not get biased
# Preprocessing lets you have the features between 1 adnd -1 for better accuracy and time complexity.
# Through svm we can also do regression (svm.SVR())

df = quandl.get('WIKI/GOOGL')

df =df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


# Updating our data set

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'  #  To be predicted

df.fillna(-99999, inplace=True)  # Replacing all nan values with outlier.

forecast_out = int(math.ceil(0.01 * len(df)))  # Testing data

df['label'] = df[forecast_col].shift(-forecast_out)

#  Defined features will be capital x and labels will be small y
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)  

'''
preprcessing.scale : To remove biasness. For example 
x = (1,40,4000,400000) is very biased towards higher 
values.GEnerally required for spatial data sets.
'''

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs = -1)   # Another Algo : svm.SVR() : N-JOBS enables to run all processing power in paraller.(Threading)
clf.fit(X_train, y_train)

with open('linearregression.pickle', 'wb') as f:  # to prevent retraining model again and again we use pickel and save it into a file.
	pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)  # gives the y predicted values

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()  # timestamp is the pandas eqivalent of python's datetime.
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix) # to convert timestamp into datetime
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  # df.loc['label'] extracts the row with the particular label

	#  df.loc[next_date] is used for putting the dates on the axis

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
