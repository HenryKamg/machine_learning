import numpy as np
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
def y(x,m,b,mu=0,sigma=1): return m*x + b + np.random.normal(mu,sigma,1)[0]
def el_pow(x,pow):
    temp = x
    for i in range(pow-1):
        temp = temp * x
    return temp
def prediction(params, x):
    pred = 0
    for i in range(len(params)):pred += params[i]*math.pow(x,i)
    return pred
#training data, with N data points
N = 101
M = 8
t = np.empty(N)
domain = np.empty(N)
domain_bound = 1.0/N
for i in range(N): domain[i] = i*domain_bound
for i in range(N): t[i] = y(x=domain[i],m=4.89,b=0.57)
#find the solution without using regularization
#design matrix, phi, N X M
phi = np.array([np.ones(N),domain, el_pow(domain,2),el_pow(domain,3),el_pow(domain,4),el_pow(domain,5),el_pow(domain,6),el_pow(domain,7)]).T
temp1 = np.linalg.inv(np.dot(phi.T,phi)) #inverse of phi.T X phi
temp2 = np.dot(temp1, phi.T)
w1 = np.dot(temp2,t) #solution
print 'w1=',w1
predicted_t = [prediction(w1,x) for x in domain]

#find the regularized solution
lam = 0.1
temp1 = np.linalg.inv(lam*np.eye(M)+np.dot(phi.T,phi))
temp2 = np.dot(temp1,phi.T)
w2 = np.dot(temp2,t)
print 'w2=',w2
predicted_t_reg = [prediction(w2,x) for x in domain]

#add some plots
plt.plot(domain,t)
plt.plot(domain,predicted_t)
plt.plot(domain,predicted_t_reg)
plt.legend(("training","un-regularized","regularized"), loc='lower right')
plt.show()
