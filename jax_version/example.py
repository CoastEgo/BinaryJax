import numpy as jnp
import jax.numpy as jnp
import jax
from uniform_model_jax import model
import time
import cProfile
jax.config.update("jax_enable_x64", True)
if 1:
    b_map=jnp.linspace(-4.0,3.0,1200)
    b=b_map[680]
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q=0.0002942995844993697;s=2.280288463020413;rho=0.0306335182013742
    q=0.1;s=1;rho=0.1
    b=0.3
    tol=1e-2
    trajectory_n=1
    alpha=alphadeg*2*jnp.pi/360
    times=jnp.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n)
    ####################
    start=time.perf_counter()
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    uniform_mag=model_uniform.get_magnifaction2(tol)
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    time_optimal=end-start
    #cProfile.run('model_uniform.debug(tol)',sort='cumulative')
    print('low_mag_mycode=',time_optimal)
    start=time.perf_counter()
    uniform_mag=model_uniform.get_magnifaction2(tol)
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    print('low_mag_mycode jit=',end-start)
