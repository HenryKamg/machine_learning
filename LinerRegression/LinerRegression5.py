import numpy as np
from matplotlib import pyplot as plt
#in order to compare between examples, set a seed in random
seed = 123456789
np.random.seed(seed)
def y(x,coefs,inc_ran = True,mu=0,sigma=0.2): 
    ans = 0
    for i in range(len(coefs)): ans += coefs[i]*math.pow(x,i)
    if inc_ran: return ans + np.random.normal(mu,sigma,1)[0]
    else: return ans
#training data, with N = 101 data points
N = 101
M = 2
t = np.empty(N)
domain = np.empty(N)
domain_bound = 3.0/N
for i in range(N): domain[i] = i*domain_bound
for i in range(N): t[i] = y(x=domain[i],coefs=[1.75, 0.25, -1.0])

#first find the standard linear regression fit using model y = phi_0 + phi_1 * x
#design matrix, phi, N X M
phi = np.array([np.ones(N),domain]).T
temp1 = np.linalg.inv(np.dot(phi.T,phi)) #inverse of phi.T X phi
temp2 = np.dot(temp1, phi.T)
w1 = np.dot(temp2,t) #solution
predicted_t = [y(x,w1,inc_ran=False) for x in domain]

#now construct the locally weighted solution
def wt(x,x_i,tau=1): return math.exp(-(x_i-x)*(x_i-x)/(2*tau*tau))
def lws(x, wt_func, td_x,td_y, Phi):
    _M = Phi.shape[1]
    _N = Phi.shape[0]
    _W = np.zeros((_N,_N))
    for i in range(_N): _W[i,i] = wt_func(x,td_x[i])
    temp1 = np.linalg.inv(np.dot(np.dot(Phi.T,_W),Phi)) #inverse of phi.T X phi
    temp2 = np.dot(np.dot(temp1,Phi.T),_W)
    w1 = np.dot(temp2,t) #local solution for parameters
    return y(x,w1,inc_ran=False)
predicted_local = [lws(x, wt, domain, t, phi) for x in domain]  
plt.plot(domain,t)
plt.plot(domain,predicted_t)
plt.plot(domain,predicted_local)
plt.legend(('training','regression', 'local regression'),loc='lower left')
