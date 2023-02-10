import numpy as np
from uniform_model_numpy import model
import sys
import time
sys.setrecursionlimit(10000)
#np.seterr(divide='ignore', invalid='ignore')
def uniform(input,times):
    t_0,t_E,q,s,alphadeg,b,rho=input
    sample_n=50
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'sample_n':sample_n})
    uniform_mag=model_uniform.mag
    return uniform_mag
def partial(input,times,delta=0.00001):
    jacobian=np.zeros((len(times),(len(input))))
    for i in range(len(input)):
        temp=np.copy(input)
        temp[i]+=input[i]*delta
        mag_diff=(uniform(temp,times)-uniform(input,times))/(delta*input[i])
        jacobian[:,i]=mag_diff
    return jacobian
input=np.array([2452848.06,61.5,0.05,0.8,30.,0.1,0.05])
t_0=2452848.06
t_E=61.5
trajectory_n=10
times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
jac=partial(input,times)
np.savetxt('jacobian.txt',jac,delimiter=',',fmt='%1.4f')