import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('Churn_Modelling.csv')
#test
#print(data)


X= data.iloc[:,3:13].values
Y = data.iloc[:,13].values


#encoder: Categorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X = ohe.fit_transform(X).toarray()
X = X[:,1:]


from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print(cm)
