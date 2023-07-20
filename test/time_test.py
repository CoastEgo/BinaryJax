import numpy as np
from ..binaryJax import model as model_jax
import jax.numpy as jnp
import VBBinaryLensing
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import time
import jax
np.seterr(divide='ignore', invalid='ignore')
def draw_colormap(data,name,normlize=colors.Normalize()):
    fig,ax=plt.subplots(figsize=(8,3))
    surf=ax.pcolormesh(traj_x,traj_y,data,cmap=cm.viridis,norm=normlize)
    ax.set_aspect('equal',adjustable='datalim')
    fig.colorbar(surf,ax=ax)
    plt.savefig('picture/'+name+'.png')
@jax.jit
def mag_gen_jax(rho,q,s,i):#t_0,b_map,t_E,rho,q,s,alphadeg,times_jax,tol,
    t_0=2452848.06;t_E=61.5
    sample_n=120
    b_map=jnp.linspace(-4.0,3.0,sample_n)
    tol=1e-2
    trajectory_n=1000;alphadeg=90
    times_jax=jnp.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n)
    uniform_mag=model_jax({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times_jax,'retol':tol})
    return uniform_mag,i
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
    sample_n=120
    tol=1e-2
    t_0=2452848.06;t_E=61.5
    times=np.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n);alphadeg=90
    times_jax=jnp.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n)
    tau=(times-t_0)/t_E
    alpha_VBBL=np.pi+alphadeg/180*np.pi
    q = 1e-3; s = 1; rho = 1e-1
    b_map=np.linspace(-4.0,3.0,sample_n)
    VBBL_mag_map=np.zeros((sample_n,trajectory_n))
    jax_map=np.zeros((sample_n,trajectory_n))
    traj_x=np.zeros((sample_n,trajectory_n))
    traj_y=np.zeros((sample_n,trajectory_n))
    ######
    start=time.perf_counter()
    #compiletime
    temp,_=mag_gen_jax(rho,q,s,0)
    print(f'compile time took: {time.perf_counter() - start:.4f}')
    for i in range(1):
        ###vbbl time
        start = time.perf_counter()
        for i in range(sample_n):
            uniform,i=mag_gen_vbbl(i)
            VBBL_mag_map[i,:]=uniform
        print(f'vbbl time took: {time.perf_counter() - start:.4f}')
        ###jax time
        start = time.perf_counter()
        for i in range(sample_n):
            uniform,i=mag_gen_jax(rho,q,s,i)
            jax_map[i,:]=uniform
        jax.block_until_ready(jax_map)
        print(f'jax time took: {time.perf_counter() - start:.4f}')
        delta2=np.abs(VBBL_mag_map-jax_map)/VBBL_mag_map
        print('error_max jax',np.max(delta2))
        '''draw_colormap(delta2,'delta',colors.LogNorm())
        draw_colormap(np.abs(VBBL_mag_map),'VBBL2.0',colors.LogNorm())
        draw_colormap(np.abs(jax_map),'mycode',colors.LogNorm())'''