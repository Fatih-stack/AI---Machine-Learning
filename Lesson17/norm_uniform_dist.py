from scipy.stats import norm, uniform, shapiro


sample_norm = norm.rvs(100, 15, size=1000)


result_norm = shapiro(sample_norm)       
sample_uniform = uniform.rvs(100, 100, size=1000) 
result_uniform = shapiro(sample_uniform)


import matplotlib.pyplot as plt 


plt.figure(figsize=(20, 8))
ax1 = plt.subplot(1, 2, 1)
ax1.set_title(f'p değeri: {result_norm.pvalue}', pad=15, fontweight='bold')


ax2 = plt.subplot(1, 2, 2)
ax2.set_title(f'p değeri: {result_uniform.pvalue}', pad=15, fontweight='bold')
ax1.hist(sample_norm, bins=20)


ax2.hist(sample_uniform, bins=20)
plt.show()
