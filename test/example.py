# MCMC_demo.py

import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import jax.numpy as jnp
import jax
from binaryJax import model
from binaryNumpy import model_ni as model_numpy
import time
import matplotlib.pyplot as plt
import VBBinaryLensing
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
if 1:
    sample_n=120
    b_map =jnp.linspace(-4.0,3.0,sample_n)
    #b=b_map[51]
    b=-2.3419516263552964
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q = 1.63882791938733e-05; s = 2.3275772945101147; rho = 0.01682812889449287
    tol=1e-3
    trajectory_n=1000
    alpha=alphadeg*2*jnp.pi/360
    times=jnp.linspace(t_0-0.*t_E,t_0+2.*t_E,trajectory_n)[3:4]
    ####################编译时间
    start=time.perf_counter()
    uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    jax.block_until_ready(uniform_mag)
    end=time.perf_counter()
    '''with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        uniform_mag=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                            'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
        jax.block_until_ready(uniform_mag)'''
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
    times=np.array(times)
    tau=(times-t_0)/t_E
    y1 = -b*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = b*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), b, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    model_uniform=model_numpy({'t_0': t_0, 'u_0': b, 't_E': t_E,
                    'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    numpy_mag=model_uniform.get_magnifaction2(tol,retol=tol)
    print('VBBL time=',(time.perf_counter()-start))
    print('VBBL time=',jax_time/(time.perf_counter()-start))
    plt.plot(times,VBBL_mag,label='VBBL')
    plt.plot(times,uniform_mag,label='binaryJax')
    plt.plot(times,numpy_mag,label='numpy')
    plt.legend()
    plt.savefig('picture/diff.png')
    delta=np.abs(np.array(uniform_mag)-VBBL_mag)/VBBL_mag
    print(np.max(delta))
    print(np.argmax(delta))
    print(np.max(np.abs(np.array(numpy_mag)-VBBL_mag)/VBBL_mag))
if 0:
    from MulensModel import caustics
    theta=np.linspace(0,2*np.pi,100)
    trajectory_l=model_uniform.trajectory_l
    zeta=model_uniform.to_centroid(model_uniform.get_zeta_l(trajectory_l[6],theta))
    fig=plt.figure(figsize=(6,6))
    plt.plot(zeta.real,zeta.imag,color='r',linewidth=0.15)
    caustic_1=caustics.Caustics(q,s)
    caustic_1.plot(5000,s=0.5)
    x,y=caustic_1.get_caustics()
    x=caustic_1.critical_curve.x
    y=caustic_1.critical_curve.y
    #plt.scatter(x,y,s=0.005)
    #curve=model_uniform.image_contour_all[6]
    plt.show()
    plt.savefig('picture/casutic.png')
    '''for k in range(len(curve)):
        cur=model_uniform.to_centroid(curve[k])
        linewidth=0.15
        plt.figure()
        plt.plot(cur.real,cur.imag,linewidth=linewidth)
        plt.axis('equal')
        plt.savefig('picture/204_'+str(k)+'.pdf',format='pdf')'''