# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 00:17:05 2023

@author: Fatih Durmaz
test population is distributed normally based on sample
"Kolmogorov-Smirnov" test and "Shapiro-Wilk" test
if p value is bigger than 0.05 -> normal  dist 
"""

from scipy.stats import uniform, shapiro, kstest, norm
import matplotlib.pyplot as plt

x = uniform.rvs(size=1000)
stat, pval = kstest(x, 'norm')
print(stat, pval)  
plt.hist(x)
plt.show()

a = norm.rvs(size=1000)
plt.hist(a)
plt.show()
stat, pval = kstest(a, 'norm')
print(f'p değeri(norm, mean:0, sd:1): {pval}')

# mean -> 100, stddev -> 15
a = norm.rvs(100, 15, size=1000)
plt.hist(a)
plt.show()
stat, pval = kstest(a, cdf=norm.cdf, args=(100, 15))
print(f'p değeri(norm, mean:100, sd:15): {pval}')

print('Shapiro-Wilk test with mean:100 sd:15')
a = norm.rvs(100, 15, size=100)
plt.hist(a)
plt.show()
stat, pval = shapiro(a)
print(f'p değeri(norm-shapiro): {pval}')

a = uniform.rvs(100, 15, size=100)
plt.hist(a)
plt.show()
stat, pval = shapiro(a)
print(f'p değeri(uniform-shapiro): {pval}')