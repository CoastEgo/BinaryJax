import numpy as np
import jax.numpy as jnp
import jax
from ..binaryJax import model
import time
import VBBinaryLensing
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if 1:
    sample_n=120
    b_map =jnp.linspace(-4.0,3.0,sample_n)
    #b=b_map[51]
    b=0.01
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q=1e-3;s=1.;rho=1e-1
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
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                            'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
        jax.block_until_ready(uniform_mag)
    time_optimal=end-start
    print('low_mag_mycode=',time_optimal)
    start=time.perf_counter()
    uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    jax_time=end-start
    #print(uniform_mag)
    print('low_mag_mycode jit=',end-start)
    start=time.perf_counter()
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=np.pi+alphadeg/180*np.pi
    VBBL.RelTol=tol
    VBBL.BinaryLightCurve
    times=np.linspace(t_0-0.*t_E,t_0+1.5*t_E,trajectory_n)
    tau=(times-t_0)/t_E
    y1 = -b*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = b*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), b, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    print('VBBL time=',(time.perf_counter()-start))
    print('VBBL time=',jax_time/(time.perf_counter()-start))
    delta=np.abs(np.array(uniform_mag)-VBBL_mag)/VBBL_mag
    print(np.max(delta))
