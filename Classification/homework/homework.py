import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


data = pd.read_excel('Iris.xls')
#print(data)


x = data.iloc[:,1:4].values
y = data.iloc[:,4:].values

# verilerin egitim ve test icin bolunmesi
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# verilerin olceklenmesi
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)
#print(y_pred)
#print(y_test)

cm = confusion_matrix(y_test, y_pred)
print("LR")
print(cm)

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("KNN")
print(cm)

svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVC')
print(cm)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('GNB')
print(cm)

dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('RFC')
print(cm)

# 7. ROC , TPR, FPR değerleri

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:, 0])

fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:, 0], pos_label='e')
print(fpr)
print(tpr)