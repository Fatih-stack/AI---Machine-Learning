import statistics 
import matplotlib.pyplot as plt

nd = statistics.NormalDist(100, 15)

# P{130 < x < 140}
result = nd.cdf(140) - nd.cdf(130)
print(result)

nd = statistics.NormalDist()

z = nd.inv_cdf(0.99)
print("Z value for 0,99 cdf value :", z)

gaussVal = nd.pdf(0)
print("Gauss value for 0 :", gaussVal)

ndArr = nd.samples(10)
print("Normal Distributed 10 number :",ndArr)

plt.hist(ndArr, bins=20)
plt.show()
