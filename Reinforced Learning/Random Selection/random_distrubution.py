import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


data = pd.read_csv('Ads_CTR_Optimisation.csv')


N = 10000
d = 10
total = 0
picks = []

for n in range(0, N):
    ad = random.randrange(d)
    picks.append(ad)
    price = data.values[n, ad]
    total = total + price

plt.hist(picks)
plt.show()