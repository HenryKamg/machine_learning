import numpy as np
import math
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
def y(x,a,b,c,mu=0,sigma=1): return a + b*math.sin(x) + c*math.cos(x) + np.random.normal(mu,sigma,1)[0]
#training data, with N data points
N = 101
M = 3
t = np.empty(N)
domain = np.empty(N)
domain_bound = 4.0*math.pi/N
for i in range(N): domain[i] = i * domain_bound
for i in range (N): t[i] = y(x=domain[i],a=1.85,b=0.57,c=4.37)
#design matrix, phi, N X M
c1 = [math.sin(x) for x in domain]
c2 = [math.cos(x) for x in domain]
phi = np.array([np.ones(N),c1,c2]).T
#find the solution
#in this case case phi.T X phi is invertible so do the folloing:
temp1 = np.linalg.inv(np.dot(phi.T,phi)) #inverse of phi.T X phi
temp2 = np.dot(temp1, phi.T)
w1 = np.dot(temp2,t) #solution
print 'w1=',w1
#assuming that phi.T X phi was not invertible we could find the pseudo inverse using the pinv function
#we expect to obtain the same solution
phi_pi = np.linalg.pinv(phi)
w2 = np.dot(phi_pi,t)
print 'w2=',w2
#compute the model predicted values for the training data domain
predicted_t = [w2[0]+w2[1]*math.sin(x)+w2[2]*math.cos(x) for x in domain]
plt.plot(domain,t)
plt.plot(domain, predicted_t)
plt.show()
