import jax.numpy as jnp
import numpy as np
from uniform_model_jax import model
from jax import jacfwd
def uniform(input,times):
    t_0,t_E,q,s,alphadeg,b,rho=input
    tol=1e-2
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    uniform_mag=model_uniform.get_magnifaction2(tol)
    return uniform_mag
input=jnp.array([2452848.06,61.5,0.05,0.8,30.,0.1,0.05])
t_0=2452848.06
t_E=61.5
trajectory_n=10
times=jnp.linspace(t_0-0.15*t_E,t_0+0.15*t_E,trajectory_n)
jacobian_fn = jacfwd(uniform, argnums=0)  # it returns the function in charge of computing jacobian
jacobian = jacobian_fn(input,times)
np.savetxt('result/jacobian_auto.txt',jacobian,delimiter=',',fmt='%1.4f')