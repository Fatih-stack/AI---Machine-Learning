# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:13:21 2023

@author: Fatih Durmaz

Parameter Estimation : Estimate mean(and/or stddev) of population based on sample
Parameter Estimation can be done as point estimate or interval(confidence interval) estimate.
"""

import numpy as np
from scipy.stats import norm, t

# confidence level (CF) : %95
sigma = 10
n = 100 # size
sample_mean = 65

samples_std = sigma / np.sqrt(n)

lower_bound = norm.ppf(0.025, sample_mean, samples_std)
upper_bound = norm.ppf(0.975, sample_mean, samples_std)

print(lower_bound, upper_bound)     # 63.04003601545995 66.95996398454005

CF = 0.99 # confidence leve increased to %99

lower_bound = norm.ppf((1 - CF) / 2, sample_mean, samples_std)
upper_bound = norm.ppf(1 - (1 - CF) / 2, sample_mean, samples_std)

print(lower_bound, upper_bound) 

for n in range(30, 105, 5):

    samples_std = sigma / np.sqrt(n)
    
    lower_bound = norm.ppf((1 - CF) / 2, sample_mean, samples_std)
    upper_bound = norm.ppf(1 - (1 - CF) / 2, sample_mean, samples_std)
    
    print(f'{n}: [{lower_bound}, {upper_bound}]') 

n = 100
samples_std = sigma / np.sqrt(n)
CF = 0.95
lower_bound, upper_bound = norm.interval(CF, sample_mean, samples_std)

print(lower_bound, upper_bound)

population = np.random.randint(0, 1_000_000_000, 1_000_000)
sigma = np.std(population)

CL = 0.99
SAMPLE_SIZE = 100

sample = np.random.choice(population, SAMPLE_SIZE)
print(f'Population Mean: {np.mean(population)}')

lower_bound, upper_bound = norm.interval(CL, np.mean(sample), sigma / np.sqrt(SAMPLE_SIZE))

print(lower_bound, upper_bound)

sample = np.array([101.93386212, 106.66664836, 127.72179427,  67.18904948,
    87.1273706 ,  76.37932669,  87.99167058,  95.16206704,
   101.78211828,  80.71674993, 126.3793041 , 105.07860807,
    98.4475209 , 124.47749601,  82.79645255,  82.65166373,
    92.17531189, 117.31491413, 105.75232982,  94.46720598,
   100.3795159 ,  94.34234528,  86.78805744,  97.79039692,
    81.77519378, 117.61282039, 109.08162784, 119.30896688,
    98.3008706 ,  96.21075454, 100.52072909, 127.48794967,
   100.96706301, 104.24326515, 101.49111644])

sample_mean = np.mean(sample)
sample_std = np.std(sample)

lower_bound = t.ppf(0.025, len(sample) - 1, sample_mean, sample_std / np.sqrt(len(sample)))
upper_bound = t.ppf(0.975, len(sample) - 1, sample_mean, sample_std / np.sqrt(len(sample)))
print(lower_bound, upper_bound)

lower_bound, upper_bound = t.interval(0.95, len(sample) - 1, sample_mean, sample_std / np.sqrt(len(sample)))
print(lower_bound, upper_bound)