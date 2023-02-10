from jax import jacfwd
import jax.numpy as np
from jax import jit
from uniform_model_jax import model
import sys
import numpy as nnp
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
input=np.array([2452848.06,61.5,0.05,0.8,30.,0.1,0.05])
t_0=2452848.06
t_E=61.5
trajectory_n=20
times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
t1=time.time()
jacobian_fn = jacfwd(uniform, argnums=0)  # it returns the function in charge of computing jacobian
jacobian = jacobian_fn(input,times)
t2=time.time()
jacobian = jacobian_fn(input,times)
t3=time.time()
mag=uniform(input,times)
t4=time.time()
print('jacobian_fist=',t2-t1)
print('jacobian_fist=',t3-t2)
print('mag=',t4-t3)
nnp.savetxt('jacobian2.txt',jacobian,delimiter=',',fmt='%1.4f')