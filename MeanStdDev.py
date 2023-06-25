import statistics
import numpy as np
import pandas as pd

a = [1, 2, 7, 8, 1, 5]
statistics.mean(a)
a = [1, 23, 56, 12, 45, 21]
statistics.median(a)

a = np.array([[1, 2, 3], [5, 6, 7], [8, 9, 10]])
np.mean(a, axis=1)
arr = np.array([1, 4, 6, 8, 4, 2, 1, 8, 9, 3, 6, 8])
mean = np.mean(arr)
np.mean(arr - mean)
np.mean(np.abs(arr - mean))

df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df
df.mean()

b = np.random.randint(1, 100, (10, 10))
df = pd.DataFrame(b)
df.median()
df.mode()

# standart deviation
def sd(a, ddof = 0):
    return np.sqrt(np.sum((a - np.mean(a)) ** 2)  / (len(a)  - ddof))
sd(arr)
np.std(arr)   # ddof -> 0
statistics.stdev(arr)  # ddof -> 1
statistics.pstdev(arr) # ddof -> 0
s = pd.Series(a)
s.std(ddof=0)

# most repeated value
a = [1, 3, 3, 4, 2, 2, 5, 2, 7, 9, 5, 3, 5, 7, 5]
statistics.mode(a)

import scipy.stats
a = np.random.randint(1, 10, (20, 10))
mr = scipy.stats.mode(a, axis=0)
mr.mode
mr.count


