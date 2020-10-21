import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('customers.csv')

X = data.iloc[:,3:].values

#k-means
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)

#finding optimum k value
result = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    result.append(kmeans.inertia_)

plt.plot(range(1,11),result)
