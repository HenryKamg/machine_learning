import numpy as np
import math
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
def y(x,a,b,c,mu=0,sigma=1): return a + b*math.sin(x) + c*math.cos(x) + np.random.normal(mu,sigma,1)[0]
N = 101
M = 3
w = np.zeros((M,1))
phi = np.empty((M,1))
eta = 0.25
#create arrays to store the values as they are generated so they can be plotted at the end
x = np.empty(N)
t = np.empty(N)
domain_bound = 4.0*math.pi/N
for i in range(N):
    x[i] = i * domain_bound
    t[i] = y(x[i],a=1.85,b=0.57,c=4.37)
    phi = np.array([[1],[math.sin(x[i])],[math.cos(x[i])]]) 
    w = w + eta*(t[i]-np.dot(w.T,phi))*phi #the learning model

print w.T
#compute the model predicted values for the training data domain
predicted_t = [w[0]+w[1]*math.sin(item)+w[2]*math.cos(item) for item in x]
plt.plot(x,t)
plt.plot(x,predicted_t)
plt.show()
