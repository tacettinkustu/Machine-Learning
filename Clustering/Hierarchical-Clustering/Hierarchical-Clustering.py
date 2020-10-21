import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values

#k-means
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
#print(kmeans.cluster_centers_)

#finding optimum k value
result = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    result.append(kmeans.inertia_)

plt.plot(range(1,11),result)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
Y_pred= kmeans.fit_predict(X)
#print(Y_pred)
plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],s=100, c='red')
plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],s=100, c='blue')
plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],s=100, c='green')
plt.scatter(X[Y_pred==3,0],X[Y_pred==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()

#HC
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_pred = ac.fit_predict(X)
print(Y_pred)

plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],s=100, c='red')
plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],s=100, c='blue')
plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],s=100, c='green')
plt.scatter(X[Y_pred==3,0],X[Y_pred==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

#dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()
