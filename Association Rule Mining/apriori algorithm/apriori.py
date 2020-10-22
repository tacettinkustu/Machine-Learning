import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

data = pd.read_csv('cart.csv', header = None)

t = []
for i in range (0,7501):
    t.append([str(data.values[i,j]) for j in range (0,20)])


rules = apriori(t,min_support=0.01, min_confidence=0.2, min_lift = 3, min_length=2)

print(list(rules))