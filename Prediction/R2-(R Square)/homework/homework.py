import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm

data = pd.read_csv('new_salaries.csv')
#print(datas)

x = data.iloc[:,2:5]
y = data.iloc[:,5:]
X = x.values
Y = y.values

print(data.corr())

#linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)


model = sm.OLS(lin_reg.predict(X),X)
print((model.fit().summary()))

#polynomial regression
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#print(lin_reg.predict([[11]]))
##print(lin_reg.predict([[6.6]]))

#print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
#print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print((model2.fit().summary()))

sc1=StandardScaler()
x_scale = sc1.fit_transform(X)
sc2=StandardScaler()
y_scale = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scale,y_scale)


#print(svr_reg.predict([[11]]))
#print(svr_reg.predict([[6.6]]))
model3 = sm.OLS(svr_reg.predict(x_scale),x_scale)
print((model3.fit().summary()))


#decision tree
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

#print(r_dt.predict([[11]]))
#print(r_dt.predict([[6.6]]))
model4 = sm.OLS(r_dt.predict(X),X)
print((model4.fit().summary()))


#random forest
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

model5 = sm.OLS(rf_reg.predict(X),X)
print((model5.fit().summary()))

#R2 values
print('-----------------------')
print('Linear R2 value')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 value')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 value')
print(r2_score(y_scale, svr_reg.predict(x_scale)))


print('Decision Tree R2 value')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 value')
print(r2_score(Y, rf_reg.predict(X)))