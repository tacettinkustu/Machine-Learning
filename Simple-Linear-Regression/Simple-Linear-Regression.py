import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('sales.csv')

#print(datas)

months = data[["Months"]]

sales = data[["Sales"]]

#print(sales)

x_train, x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.transform(y_test)

lr = LinearRegression()
lr.fit(x_train,y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Sales Graphic")

plt.xlabel("Months")
plt.ylabel("Sales")