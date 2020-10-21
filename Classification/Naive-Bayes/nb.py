import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv('data.csv')
#print(data)


x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values


x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#logistic regression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
#print(y_pred)
#print(y_test)

#confusion matrix
cm = confusion_matrix(y_test,y_pred)
#print(cm)


#K-NN
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
#print(cm)


#support-vektor-machine
svc = SVC(kernel="rbf")
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
#print(cm)


#naive bayes
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

