import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


datas = pd.read_csv('datas.csv')
#print(datas)


weight = datas[["weight"]]
height = datas[["height"]]
height_weight = datas [["height","weight"]]
#print(height_weight)


imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
Age = datas.iloc[:,1:4].values
#print((Age))
imputer = imputer.fit(Age[:,1:4])
#print(Age)

country = datas.iloc[:,0:1].values
#print(country)
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
#print(country)
ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
#print(country)


sex = datas.iloc[:,-1:].values
le = preprocessing.LabelEncoder()
sex[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
sex = ohe.fit_transform(sex).toarray()
#print(sex)


result = pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
#print(result)
result2 = pd.DataFrame(data=Age,index=range(22),columns=["height","weight","age"])
#print(result2)

sex1 = datas.iloc[:,-1].values
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