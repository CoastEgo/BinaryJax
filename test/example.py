# MCMC_demo.py

import sys
import os
global numofchains
numofchains = 10
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=%d'%numofchains
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
## write a warp function to calculate time of a function
def timeit(f):
    ### calculate the mean and std of the time
    def timed(*args,**kw):
        ts=time.perf_counter()
        result=f(*args,**kw)
        te=time.perf_counter()
        print(f'{f.__name__} compile time={te-ts}')
        alltime=[]
        for i in range(100):
            ts=time.perf_counter()
            result=f(*args,**kw)
            te=time.perf_counter()
            alltime.append(te-ts)
        alltime=np.array(alltime)
        print(f'{f.__name__} time={np.mean(alltime)}+/-{np.std(alltime)}')
        return result
    return timed
def model_test(parms):
    mag=model(parms)
    jax.block_until_ready(mag)
    return mag
def model_pmap(parms,i):
    parms['times']=parms['times'][:,i]
    return model(parms)
@timeit
def jacobian_test(grad_fun,times,tol):
    jac=grad_fun(8280.094505,0.121803,39.824343,0.002589090,10**(-0.076558),10**(0.006738),56.484044,times,tol)
    jax.block_until_ready(jac)
    return jac
@timeit
def VBBL_test_cloop(parms):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=np.pi+parms['alpha_deg']/180*np.pi
    VBBL.RelTol=parms['retol']
    VBBL.BinaryLightCurve
    times=np.array(parms['times'])
    tau=(times-parms['t_0'])/parms['t_E']
    y1 = -parms['u_0']*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = parms['u_0']*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(parms['s']), np.log(parms['q']), parms['u_0'], alpha_VBBL, np.log(parms['rho']), np.log(parms['t_E']), parms['t_0']]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return VBBL_mag
@timeit
def VBBL_test_pythonloop(parms):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=np.pi+parms['alpha_deg']/180*np.pi
    VBBL.RelTol=parms['retol']
    times=np.array(parms['times'])
    tau=(times-parms['t_0'])/parms['t_E']
    y1 = -parms['u_0']*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = parms['u_0']*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    VBBL_mag = []
    for i in range(len(times)):
        VBBL_mag.append(VBBL.BinaryMag2(parms['s'],parms['q'],y1[i],y2[i],parms['rho']))
    return VBBL_mag
def numpy_test(parms):
    mag=model_numpy(parms).get_magnifaction2(parms['retol'],retol=parms['retol'])
    return mag
def time_test():
    sample_n=120
    b_map =jnp.linspace(-4.0,3.0,sample_n)
    #b=b_map[51]
    b=0.121803
    t_0=8280.094505;t_E=39.824343;alphadeg=56.484044
    q = 10**(0.006738); s = 10**(-0.076558); rho = 10**(-2.589090)
    tol=1e-3
    trajectory_n=1000
    alpha=alphadeg*2*jnp.pi/360
    times=jnp.linspace(t_0-2.*t_E,t_0+2*t_E,trajectory_n)
    ####################编译时间
    parm={'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol}
    parm2=parm.copy()
    #parm2['times']=jnp.reshape(parm2['times'],(-1,numofchains),order='C')
    uniform_mag=timeit(model_test)(parm)
    '''with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        ## disable jit
        with jax.disable_jit():  
            model_test(parm)'''
    ## compile time
    def sum_fun(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol):
        #times=jnp.reshape(times,(-1,numofchains),order='C')
        dic={'t_0': t_0, 'u_0': u_0, 't_E': t_E,
            'rho': rho, 'q': q, 's': s, 'alpha_deg': alpha_deg,'times':times,'retol':retol}
        mag=model_test(dic)
        #mag=jax.pmap(model_pmap,in_axes=(None,0))(dic,jnp.arange(jax.device_count()))
        #mag=jnp.reshape(mag,(trajectory_n,),order='F')
        return jnp.sum(mag)
    #### calculate the jacbian matrix and print time
    #grad_fun=jax.jacfwd(sum_fun,argnums=(0,1,2,3,4,5,6))
    #jac=timeit(jacobian_test)(grad_fun,times,tol)
    #print(jac)
    VBBL_mag=VBBL_test_cloop(parm)
    VBBL_mag2=VBBL_test_pythonloop(parm)
    model_uniform=model_numpy({'t_0': t_0, 'u_0': b, 't_E': t_E,
                    'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    numpy_mag=model_uniform.get_magnifaction2(tol,retol=tol)
    numpy_mag=numpy_test(parm)
    plt.plot(times,VBBL_mag,label='VBBL')
    plt.plot(times,uniform_mag,label='binaryJax')
    plt.plot(times,numpy_mag,label='numpy')
    plt.legend()
    plt.savefig('picture/diff.png')
    delta=np.abs(np.array(uniform_mag)-VBBL_mag)/VBBL_mag
    print(np.max(delta))
    print(np.argmax(delta))
    print(np.max(np.abs(np.array(numpy_mag)-VBBL_mag)/VBBL_mag))
def contour_plot(parms):
    ## function of numpy version , you can use it to plot the image contour 
    # and check whether the caustic crossing event is correct
    from MulensModel import caustics
    model_uniform=model_numpy(parms)
    mag=model.get_magnifaction2(parms['retol'],retol=parms['retol'])
    curve=model.image_contour_all
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
    plt.savefig('picture/casutic.png')
    '''for k in range(len(curve)):
        cur=model_uniform.to_centroid(curve[k])
        linewidth=0.15
        plt.figure()
        plt.plot(cur.real,cur.imag,linewidth=linewidth)
        plt.axis('equal')
        plt.savefig('picture/204_'+str(k)+'.pdf',format='pdf')'''
def caustic_test():
    ##test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py
    ### you may need to change Max_array_length in binaryJax/model_jax.py to avoid the failure caused by fixed array length
    from MulensModel import caustics
    import MulensModel as mm
    from binaryJax import contour_integrate,to_centroid,to_lowmass
    from jax import lax,random
    import binaryJax
    def test_extend_sorce(rho,q,s,retol=1e-3):
        caustic_1=caustics.Caustics(q,s)
        x,y=caustic_1.get_caustics()
        z_centeral=jnp.array(jnp.array(x)+1j*jnp.array(y))
        ## random change the position of the source
        key = random.PRNGKey(42)
        key, subkey1, subkey2 = random.split(key, num=3)
        phi = jax.random.uniform(subkey1, z_centeral.shape, minval=-np.pi, maxval=np.pi)
        r = random.uniform(subkey2, z_centeral.shape, minval=0., maxval=2*rho)
        z_centeral = z_centeral + r*np.exp(1j*phi)
        fig,(ax,ax2)=plt.subplots(2,1,figsize=(6,8))
        ax.scatter(z_centeral.real,z_centeral.imag,s=0.5,color='orange',label='source')
        circle = plt.Circle((z_centeral.real[0], z_centeral.imag[0]), rho, fill=False,color='green')  # 创建一个圆，中心在 (z.real, z.imag)，半径为 rho
        ax.add_patch(circle)  # 将圆添加到坐标轴上
        ax.scatter(x,y,s=0.5,color='deepskyblue',label='caustic')
        ax.set_aspect('equal')
        ### change the coordinate system
        z_lowmass=to_lowmass(s,q,z_centeral)
        trajectory_n=z_centeral.shape[0]
        ### change the coordinate system
        m1=1/(1+q)
        m2=q/(1+q)
        VBBL=VBBinaryLensing.VBBinaryLensing()
        VBBL.RelTol=retol
        VBBL_mag=[]
        start=time.time()
        for i in range(trajectory_n):
            VBBL_mag.append(VBBL.BinaryMag2(s,q,z_centeral.real[i],z_centeral.imag[i],rho))
        print(f'average VBBL time={(time.time()-start)/trajectory_n}')
        VBBL_mag=np.array(VBBL_mag)
        Jax_mag=[]
        ## compile time
        mag=contour_integrate(rho,s,q,m1,m2,z_lowmass[0],retol,epsilon_rel=retol)[0][-3][0]
        ## real time
        start=time.time()
        for i in range(trajectory_n):
            mag=contour_integrate(rho,s,q,m1,m2,z_lowmass[i],retol,epsilon_rel=retol)[0][-3][0]
            Jax_mag.append(mag)
        jax.block_until_ready(Jax_mag)
        print(f'average Jax time={(time.time()-start)/trajectory_n}')
        Jaxmag=np.array(Jax_mag)
        print(f'rho={rho},max error={np.max(np.abs(Jaxmag-VBBL_mag)/VBBL_mag)}')
        ax2.plot(np.abs(Jaxmag-VBBL_mag)/VBBL_mag)
        #ax2.plot(Jax_mag,color='r',label='binaryJax')
        #ax2.plot(VBBL_mag,color='b',label='VBBL')
        ## set log scale
        ax2.set_yscale('log')
        ax2.set_ylabel('relative error')
        plt.savefig(f'picture/extendtest_{rho}.png')
    rho=[1e-2]
    for i in range(len(rho)):
        test_extend_sorce(rho[i],1e-3,1)
def jacobian_test(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol):
    def model_list(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol):
        dic={'t_0': t_0, 'u_0': u_0, 't_E': t_E,
            'rho': rho, 'q': q, 's': s, 'alpha_deg': alpha_deg,'times':times,'retol':retol}
        mag=model(dic)
        return mag
    grad_fun=jax.jacfwd(model_list,argnums=(0,1,2,3,4,5,6))
    jacobian=jnp.array(grad_fun(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol))
    name=['t_0','u_0','t_E','rho','q','s','alpha_deg']
    plt.figure(figsize=(10,12))
    dic={'t_0': t_0, 'u_0': u_0, 't_E': t_E,
    'rho': rho, 'q': q, 's': s, 'alpha_deg': alpha_deg,'times':times,'retol':retol}
    mag=model(dic)
    plt.subplot(8,1,1)
    plt.plot(times,mag,label='light curve')
    plt.legend()
    for i in range(jacobian.shape[0]):
        plt.subplot(8,1,i+2)
        plt.plot(times,jnp.abs(jacobian[i,:]),label=name[i])
        plt.legend()
        plt.ylim(1e-5,1e5)
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig('picture/jacobian_caustic.png')
if __name__=='__main__':
    b=0.121803
    t_0=8280.094505;t_E=39.824343;alphadeg=56.484044
    q = 10**(0.006738); s = 10**(-0.076558); rho = 10**(-2.589090)
    tol=1e-3
    trajectory_n=1000
    alpha=alphadeg*2*jnp.pi/360
    times=jnp.linspace(t_0-1.*t_E,t_0+1*t_E,trajectory_n)
    ####################编译时间
    parm={'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol}
    jacobian_test(**parm)