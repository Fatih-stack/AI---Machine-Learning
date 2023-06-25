# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:01:57 2023

@author: Fatih Durmaz

Lesson 9
Normal distribution :
   - Cumulative distribution function
   - Percent Point function
   - Probability Density function
continous uniform distribution
cdf, ppf, pdf and rvs methods
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
mean : 100, sd : 15
"""

resultCDF = norm.cdf([100, 130, 140], 100, 15)
print("mean : 100, sd : 15")
print("cdf for 100 : ", resultCDF[0])
print("cdf for 130 : ", resultCDF[1])
print("cdf for 140 : ", resultCDF[2])

print()
print("PPF is inverse of CDF")
resultPPF = norm.ppf([0.50, 0.68, 0.95], 100, 15)
print("value for 0.50 cdf ->", resultPPF[0])
print("value for 0.68 cdf ->", resultPPF[1])
print("value for 0.95 cdf ->", resultPPF[2])

x = np.linspace(40, 160, 1000)
y = norm.pdf(x, 100, 15) # y value on gauss curve

plt.plot(x, y)

x = np.full(200, 100)       # 200 tane 100'lerden oluşan dizi
yend = norm.pdf(100, 100, 15)
y = np.linspace(0, yend, 200)
plt.plot(x, y, linestyle='--')

plt.show()

norm.rvs(100, 15, 10000)    # rvs -> generate random number
plt.hist(x, bins=20)
plt.show()

"""
P{mu - sigma < x < mu + sigma} -> 0.68 in normal distribution
"""

result = norm.cdf(1) - norm.cdf(-1)
print("P{mu - sigma < x < mu + sigma} ->", result)

x = np.linspace(-5, 5, 1000)
y = norm.pdf(x)

axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-1, 1, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

"""
P{mu - 2 * sigma < x < mu + 2 * sigma} -> 0.95 in normal distribution
"""

result = norm.cdf(2) - norm.cdf(-2)
print("P{mu - 2 * sigma < x < mu + 2 * sigma} ->", result)

axis.set_xticks(range(-4, 5))
axis.text(2, 0.3, f'{result:.3f}', fontsize=14, fontweight='bold')

plt.plot(x, y)

x = np.linspace(-2, 2, 1000)
y = norm.pdf(x)
plt.fill_between(x, y)
axis.arrow(2.5, 0.25, -2, -0.1, width=0.0255)

plt.show()

"""
continous uniform distribution default a = 0, b = 1
"""
from scipy.stats import uniform

result = uniform.cdf(0.5)
print(result)       # 0.5

result = uniform.cdf(0.8)
print(result)       # 0.8

result = uniform.pdf([0.1, 0.2, 0.3])
print(result)       # [1. 1. 1.]

result = uniform.ppf([0.1, 0.2, 0.3])
print(result)       # [0.1 0.2 0.3]