import numpy as np
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
def y(x,m,b,mu=0,sigma=1): return m*x + b + np.random.normal(mu,sigma,1)[0]
#training data, with N data points
N = 101
M = 2
t = np.empty(N)
domain_bound = 1.0/N
domain = np.empty(N)
for i in range(N): domain[i] = i*domain_bound
for i in range(N): t[i] = y(x=domain[i],m=4.89,b=0.57)
#design matrix, phi, N X M
phi = np.array([np.ones(N),domain]).T
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
predicted_t = [w2[0]+w2[1]*x for x in domain]
plt.plot(domain,t)
plt.plot(domain,predicted_t)
plt.show()
