import numpy as np
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
alpha = 0.4
beta = 5.0
def y(x,coefs, mu=0, sigma=1.0/beta): 
    ans = 0
    for i in range(len(coefs)): ans += coefs[i]*math.pow(x,i)
    return ans + np.random.normal(mu,sigma,1)[0]
    
#training data, with N = 101 data points
N = 101
M = 4
t = np.empty(N)
domain = np.empty(N)
domain_bound = 3.0/N
for i in range(N): domain[i] = i*domain_bound
for i in range(N): t[i] = y(x=domain[i],coefs=[1.75, 0.25, -1.0])

#Let's assume that we want to fit a 3rd order polynomial to the data even though we know its a second order
#polynomial. Given the Bayesain approach, we should see the so that the unecessary terms are damped out. We have 
#y = phi_0 + phi_1 * x + phi_2 x^2 + phi_3 x^4
#design matrix, phi, N X M where N = 101 and M = 4
d2 = domain * domain
phi = np.array([np.ones(N),domain, d2, d2 * domain]).T
alphaI = alpha * np.eye(M)
SN = np.linalg.inv(alphaI + beta * np.dot(phi.T,phi)) #posterior variance
mN = beta * np.dot(np.dot(SN, phi.T), t)
point_estimates = [np.dot(mN, phi[i]) for i in range(N)]
uncertain_t = [1.0/beta + np.dot(np.dot(phi[i].T, SN), phi[i]) for i in range(N)]
plt.plot(domain,t)
plt.errorbar(domain,point_estimates, uncertain_t, ecolor = "red")
plt.legend(('training','Bayes'),loc='lower left')
