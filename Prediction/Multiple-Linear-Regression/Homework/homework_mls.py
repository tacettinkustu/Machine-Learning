import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LinearRegression

datas = pd.read_csv('tennis.csv')
#print(datas)


outlook = datas[["outlook"]]
temperature = datas[["temperature"]]
humidity = datas [["humidity"]]
windy = datas [["windy"]]
play = datas [["play"]]

"""
outlook = datas.iloc[:,0:1].values
#print(outlook)
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(datas.iloc[:,0])
#print(outlook)
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
#print(outlook)


windy = datas.iloc[:,3:4].values
print(windy)
le = preprocessing.LabelEncoder()
windy[:,0] = le.fit_transform(windy.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()


play = datas.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
play[:,-1] = le.fit_transform(play.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()
#print(play)
"""

datas2 = datas.apply(preprocessing.LabelEncoder().fit_transform)
c = datas.iloc[:,0:1].values
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(outlook).toarray()
#print(c)


air = pd.DataFrame(data=c,index=range(14),columns=["o","r","s"])
last_datas=pd.concat([air,datas.iloc[:,1:3]],axis=1)
last_datas = pd.concat([datas2.iloc[:,-2:],last_datas],axis=1)


x_train, x_test,y_train,y_test = train_test_split(last_datas.iloc[:,:-1],last_datas.iloc[:,-1:],test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

data = last_datas.iloc[:,:-1]
data2 = last_datas.iloc[:,-1:]


#backward elemination
X = np.append(arr = np.ones((14,1)).astype(int),values = data ,axis = 1)
X_l = data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(data2,X_l).fit()
#print(model.summary())

X = np.append(arr = np.ones((14,1)).astype(int),values = data ,axis = 1)
X_l = data.iloc[:,[1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(data2,X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)