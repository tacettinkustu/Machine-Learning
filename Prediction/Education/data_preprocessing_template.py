#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#2.veri onisleme
#2.1.veri yukleme
datas = pd.read_csv('data.csv')
#pd.read_csv("data.csv")
#test
print(datas)

x = datas.iloc[:,1:4].values #bağımsız değişkenler
y = datas.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin transformu
from sklearn import preprocessing

country = datas.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

country[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder()

country = ohe.fit_transform(country).toarray()

print(country)

#missing datas

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

















