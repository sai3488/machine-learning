import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 5
var = 1
sigma = np.sqrt(var)

points = np.linspace(mu - 5*sigma, mu + 5*sigma, 10)
plt.plot(points,norm.pdf(points,mu,sigma), 'go-')
plt.show()

plt.figure(figsize=(30,60))
mu = [0,1,2,3,4,5,6,7,8,9,10]
def plot_graph(mu,sigma,i):
  plt.subplot(12, 4, i+1)
  plt.plot(points, norm.pdf(points, mu, 1))
  plt.plot(points, norm.pdf(points, 5, 1), 'go')
  plt.title("Mean = " + str(mu) + ", Variance = 1")

for i in range(11):
  plot_graph(i,sigma,i)
  
plt.show()

a = []
def plot_graph(mu,sigma):
  s=0
  for i in points:
    s += np.log(norm.pdf(i,mu,sigma))
  a.append(s)

for i in mu:
  plot_graph(i,1)

# print(a)
plt.plot(mu,a)
plt.plot(mu,a,'go')
plt.xlabel("Mean")
plt.ylabel("Log_Likelihood")
plt.show()
