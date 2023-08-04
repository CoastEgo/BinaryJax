import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=150'
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from binaryNumpy import model as model_numpy
import VBBinaryLensing
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from multiprocessing import Pool
import multiprocessing as mp
import time
np.seterr(divide='ignore', invalid='ignore')
from binaryJax import model as model_jax
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
def draw_colormap(data,name,normlize=colors.Normalize()):
    fig,ax=plt.subplots(figsize=(8,3))
    surf=ax.pcolormesh(traj_x,traj_y,data,cmap=cm.viridis,norm=normlize)
    ax.set_aspect('equal',adjustable='datalim')
    fig.colorbar(surf,ax=ax)
    plt.savefig('picture/'+name+'.png')
def mag_gen_numpy(i):
    model_uniform=model_numpy({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    mag_numpy=model_uniform.get_magnifaction2(tol,retol=tol)
    return (mag_numpy,i)
def mag_gen_jax(rho,q,s,i):#t_0,b_map,t_E,rho,q,s,alphadeg,times_jax,tol,
    t_0=2452848.06;t_E=61.5
    b_map=jnp.linspace(-4.0,3.0,sample_n)
    tol=1e-3
    trajectory_n=1000;alphadeg=90
    times_jax=jnp.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n)
    uniform_mag=model_jax({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times_jax,'retol':tol})
    return uniform_mag
def mag_gen_vbbl(i):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol=tol
    VBBL.BinaryLightCurve
    y1 = -b_map[i]*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = b_map[i]*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), b_map[i], alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return (VBBL_mag,i)
if __name__=="__main__":
    trajectory_n=1000
    sample_n=1200
    tol=1e-3
    t_0=2452848.06;t_E=61.5
    times=np.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n);alphadeg=90
    times_jax=jnp.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n)
    tau=(times-t_0)/t_E
    alpha_VBBL=np.pi+alphadeg/180*np.pi
    q=1e-3;s=1;rho=1e-3
    b_map=jnp.linspace(-4.0,3.0,sample_n)
    VBBL_mag_map=np.zeros((sample_n,trajectory_n))
    numpy_map=np.zeros((sample_n,trajectory_n))
    jax_map=np.zeros((sample_n,trajectory_n))
    traj_x=np.zeros((sample_n,trajectory_n))
    traj_y=np.zeros((sample_n,trajectory_n))
    ######
    process_number=150
    start=time.monotonic()
    temp=jax.pmap(mag_gen_jax)(jnp.full((process_number,),rho),jnp.full((process_number,),q),jnp.full((process_number,),s),jnp.arange(process_number))
    #compiletime)
    print(f'compile time took: {time.monotonic() - start:.1f}')
    for i in range(1000):
        #print(i)
        q = 10**(np.random.uniform(-6.,0.))
        s = np.random.uniform(0.1,4.)
        rho = 10**(np.random.uniform(-3.,-1.))
        print(f'parameter:q = {q}; s = {s}; rho = {rho}')#'''
        ###numpy time
        start = time.monotonic()
        with Pool(processes=process_number) as pool:
            for uniform,i in pool.imap(mag_gen_numpy,range(sample_n)):
                numpy_map[i,:]=uniform
            pool.close()
            pool.join()
        #print(f'numpy time took: {time.monotonic() - start:.1f}')
        ###vbbl time
        start = time.monotonic()
        with Pool(processes=process_number) as pool:
            for uniform,i in pool.imap(mag_gen_vbbl,range(sample_n)):
                VBBL_mag_map[i,:]=uniform
            pool.close()
            pool.join()
        '''for i in range(sample_n):
            uniform,i=mag_gen_vbbl(i)
            VBBL_mag_map[i,:]=uniform'''
        #print(f'vbbl time took: {time.monotonic() - start:.1f}')
        ###jax time
        '''start = time.monotonic()
        for i in range(sample_n):
            jax_map[i,:]=mag_gen_jax(i)'''
        start = time.monotonic()
        all_nodes=jnp.arange(sample_n)
        outputs=[]
        for i in range(sample_n//process_number):
            slic=all_nodes[process_number*i:process_number*(i+1)]
            outputs.append(jax.pmap(mag_gen_jax)(jnp.full((process_number,),rho),jnp.full((process_number,),q),jnp.full((process_number,),s),slic))
        #slic=all_nodes[process_number*(i+1):]
        #outputs.append(jax.pmap(mag_gen_jax)(slic))
        jax_map=jnp.concatenate(outputs,axis=0)
        #print(f'jax time took: {time.monotonic() - start:.1f}')
        delta1=np.abs(VBBL_mag_map-numpy_map)/VBBL_mag_map
        delta2=np.abs(VBBL_mag_map-jax_map)/VBBL_mag_map
        if jnp.max(delta2)>3*tol:
            print('error_max numpy',jnp.max(delta1))
            print('error_max jax',jnp.max(delta2))
            b_i=jnp.where(delta2==jnp.max(delta2))[0][0]
            print(b_map[b_i])
        #draw_colormap(delta,'delta',colors.LogNorm())
        #draw_colormap(np.abs(VBBL_mag_map),'VBBL2.0',colors.LogNorm())
        #draw_colormap(np.abs(mycode_map),'mycode',colors.LogNorm())'''