import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import matrix_rank, inv, solve 

x =np.array([1, 2, 3 ])
t = np.array([1.2, 1.9, 3.2])
beta = 1

#maximizing Loglikelihood for w0 and w1 and then solving those equations

A = t.sum()
B = x.sum()
E = len(x)
C = (x*t).sum()
D = (np.power(x, 2)).sum()

w1 = (C*E - A*B)/(E*D - np.power(B, 2))
w0 = ((A-B)*w1)/E

slope, intercept = w1, w0

abline_values = [slope * i + intercept for i in x]
print( w1, w0)

plt.scatter(x, t)
plt.plot(x, abline_values, 'b')
plt.show()
