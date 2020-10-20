import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LinearRegression


data = pd.read_csv('data.csv')
#print(datas)


weight = data[["weight"]]
height = data[["height"]]
height_weight = data [["height","weight"]]
#print(height_weight)


imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
Age = data.iloc[:,1:4].values
#print((Age))
imputer = imputer.fit(Age[:,1:4])
#print(Age)

country = data.iloc[:,0:1].values
#print(country)
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(data.iloc[:,0])
#print(country)
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
#print(country)


sex = data.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
sex[:,-1] = le.fit_transform(data.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
sex = ohe.fit_transform(sex).toarray()
#print(sex)


result = pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
#print(result)
result2 = pd.DataFrame(data=Age,index=range(22),columns=["height","weight","age"])
#print(result2)

sex1 = data.iloc[:,-1].values
#print((sex1))

result3 = pd.DataFrame(data=sex[:,:1],index=range(22),columns=["sex"])
#print(result3)


res = pd.concat([result,result2],axis=1)
#print(res)
res2 = pd.concat([res,result3],axis=1)
#print((res2))


x_train, x_test,y_train,y_test = train_test_split(res,result3,test_size=0.33, random_state=0)


sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

height = res2.iloc[:,3:4].values
#print(weight)

left = res2.iloc[:,:3]
right = res2.iloc[:,4:]

data1 = pd.concat([left,right],axis=1)


x_train, x_test,y_train,y_test = train_test_split(data1,height,test_size=0.33, random_state=0)
regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)


"""
this section is very important for showing 
"Backward" elemination
"""
X = np.append(arr = np.ones((22,1)).astype(int),values = data1 ,axis = 1)

X_l = data1.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
#print(model.summary())

X_l = data1.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
#print(model.summary())

X_l = data1.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())