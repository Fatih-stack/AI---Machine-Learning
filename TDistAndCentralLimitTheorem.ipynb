# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:02:05 2023

@author: Fatih Durmaz
t distribution 
    - standart deviation depends on degree of freedom
    - stddev -> sigma = karekök(df (df - 2))
    - if df(degree of freedom) >= 30 it is so similar to normal dist.
    - generally mean -> 0, stddev -> 1
from scipy.stats import t -> as below
    its funcntions params(generally) -> x values, df, mean, stddev
Compare t dist and standart normal dist
t distributions with various degree of freedom values

inferential statistics -> central limit theorem
According to this theorem,
    the means of the certain size samples which are taken from a population
    are normally distributed.
    The mean of the means of graph samples from the population 
    is similar to the mean of the population.
    stddev of means of samples -> stddev of population / sqrt(n)
    stddev of means of samples is also called as standard error.
"""

from scipy.stats import norm,  t
import numpy as np
import matplotlib.pyplot as plt

DF = 3

x = np.linspace(-5, 5, 1000)

fig = plt.gcf()
fig.set_size_inches((10, 8))

axis = plt.gca()
axis.set_ylim(-0.5, 0.5)
axis.spines['left'].set_position('center')
axis.spines['bottom'].set_position('center')
axis.spines['top'].set_color(None)
axis.spines['right'].set_color(None)

y_norm = norm.pdf(x)
y_t = t.pdf(x, DF)

plt.plot(x, y_norm, color='blue')
plt.plot(x, y_t, color='red')

plt.legend(['Standart Normal Distribution', 't Distribution'], fontsize=14)

plt.show()

"""
t distributions with various degree of freedom values
"""

plt.plot(x, y_norm, color='blue')

df_info = [(2, 'red'), (5, 'green'), (10, 'black')]

for df, color in df_info:
    y_t = t.pdf(x, df)
    plt.plot(x, y_t, color=color)
    
plt.legend(['Standart Normal Dağılım'] + [f'{t[0]} Serbestlik Derecesi' for t in df_info], fontsize=14)

plt.show()

"""
inferential statistics -> central limit theorem
"""

x = np.random.randint(0, 1_000_000_000, 1_000_000)
samples = np.random.choice(x, (10_000, 50))
samples_means = np.mean(samples, axis=1)

plt.hist(samples_means, bins=50)
population_mean = np.mean(x)
sample_means_mean = np.mean(samples_means)
population_std = np.std(x)
sample_means_std = np.std(samples_means)

plt.show()

print(f'Mean of Population: {population_mean}')
print(f'Mean of means of samples: {sample_means_mean}')
print(f'StdDev of population / root (50) = {population_std / np.sqrt(50)}')
print(f'StdDev of means of samples: {sample_means_std}')



