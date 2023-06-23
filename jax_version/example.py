import numpy as jnp
import jax.numpy as jnp
import jax
from uniform_model_jax import model
import time
import cProfile
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if 1:
    sample_n=120
    b_map =jnp.linspace(-4.0,3.0,sample_n)
    b=b_map[51]
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q=1e-3;s=1;rho=0.001
    tol=1e-2
    trajectory_n=100
    alpha=alphadeg*2*jnp.pi/360
    times=jnp.linspace(t_0-0.*t_E,t_0+1.5*t_E,trajectory_n)
    ####################编译时间
    start=time.perf_counter()
    uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    '''with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        uniform_mag=model_uniform.get_magnifaction2(tol,retol=tol)
        jax.block_until_ready(uniform_mag)'''
    time_optimal=end-start
    print('low_mag_mycode=',time_optimal)
    start=time.perf_counter()
    uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    print(uniform_mag)
    print('low_mag_mycode jit=',end-start)
