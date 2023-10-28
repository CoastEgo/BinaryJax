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
if 0:##test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py
    ### you may need to change Max_array_length in binaryJax/model_jax.py to avoid the failure caused by fixed array length
    from MulensModel import caustics
    import MulensModel as mm
    from binaryJax import contour_integrate,to_centroid,to_lowmass
    from jax import lax,random
    import binaryJax
    def test_extend_sorce(rho,q,s,retol=1e-3,Max_array_length=500):
        caustic_1=caustics.Caustics(q,s)
        x,y=caustic_1.get_caustics()
        z_centeral=jnp.array(jnp.array(x)+1j*jnp.array(y))
        ## random change the position of the source
        key = random.PRNGKey(42)
        key, subkey1, subkey2 = random.split(key, num=3)
        phi = jax.random.uniform(subkey1, z_centeral.shape, minval=-np.pi, maxval=np.pi)
        r = random.uniform(subkey2, z_centeral.shape, minval=0., maxval=2*rho)
        z_centeral = z_centeral + r*np.exp(1j*phi)
        ### change the coordinate system
        z_lowmass=to_lowmass(s,q,z_centeral)
        trajectory_n=z_centeral.shape[0]
        ### change the coordinate system
        m1=1/(1+q)
        m2=q/(1+q)

        VBBL=VBBinaryLensing.VBBinaryLensing()
        VBBL.RelTol=retol
        VBBL_mag=[]
        for i in range(trajectory_n):
            VBBL_mag.append(VBBL.BinaryMag2(s,q,z_centeral.real[i],z_centeral.imag[i],rho))
        VBBL_mag=np.array(VBBL_mag)
        binaryJax.Max_array_length=Max_array_length
        Jax_mag=[]
        for i in range(trajectory_n):
            Jax_mag.append(contour_integrate(rho,s,q,m1,m2,z_lowmass[i],retol,epsilon_rel=retol)[-3][0])
        Jaxmag=np.array(Jax_mag)
        print(f'rho={rho},max error={np.max(np.abs(Jaxmag-VBBL_mag)/VBBL_mag)}')
        plt.figure()
        plt.plot(np.abs(Jaxmag-VBBL_mag)/VBBL_mag)
        ## set log scale
        plt.yscale('log')
        plt.ylabel('relative error')
        plt.savefig(f'picture/extendtest_{rho}.png')
    rho=[1,1e-1,1e-2,1e-3]
    for i in range(len(rho)):
        test_extend_sorce(rho[i],1e-5,1)