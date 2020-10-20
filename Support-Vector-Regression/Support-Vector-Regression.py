import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

datas = pd.read_csv('salaries.csv')
#print(datas)

x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]
X = x.values
Y = y.values

#linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")


#polynomial regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

poly_reg = PolynomialFeatures(degree=8)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

#print(lin_reg.predict([[11]]))
##print(lin_reg.predict([[6.6]]))

#print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
#print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


sc1=StandardScaler()
x_scale = sc1.fit_transform(X)
sc2=StandardScaler()
y_scale = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scale,y_scale)
plt.scatter(x_scale,y_scale,color='red')
plt.plot(x_scale,svr_reg.predict(x_scale),color='blue')


print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))









