import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


data = pd.read_csv('Ads_CTR_Optimisation.csv')

#random selection
"""
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
"""

"""
#UCB
N = 10000
d = 10
#Ri(n)
price = [0] * d
#Ni(n)
click = [0] * d
total = 0
pick = []
for n in range(1,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if(click[i] > 0):
            ortalama = price[i] / click[i]
            delta = math.sqrt(3/2* math.log(n)/click[i])
            ucb = ortalama + delta
        else:
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    pick.append(ad)
    click[ad] = click[ad]+ 1
    price1 = data.values[n,ad]
    price[ad] = price[ad]+ price1
    total = total + price1
print('Total Price:')
print(total)

plt.hist(pick)
plt.show()
"""

#thompson
N = 10000
d = 10
#Ri(n)
price = [0] * d
#Ni(n)
total = 0
pick = []
ones = [0]*d
zeros = [0]*d
for n in range(1,N):
    ad = 0
    max_th = 0
    for i in range(0,d):
        ranbeta = random.betavariate(ones[i]+1,zeros[i]+1)
        if(ranbeta > max_th):
            max_th = ranbeta
            ad = i
    pick.append(ad)
    price1 = data.values[n,ad]
    if price1 == 1:
        ones[ad] = ones[ad]+1
    else:
        zeros[ad] = zeros[ad] +1
    total = total + price1
print('Total Price:')
print(total)

plt.hist(pick)
plt.show()