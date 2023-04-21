import numpy as np
from uniform_model_numpy import model
import VBBinaryLensing
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from multiprocessing import Pool
import multiprocessing as mp
import time
import traceback
np.seterr(divide='ignore', invalid='ignore')
def draw_colormap(data,name,normlize=colors.Normalize()):
    fig,ax=plt.subplots(figsize=(8,3))
    surf=ax.pcolormesh(traj_x,traj_y,data,cmap=cm.viridis,norm=normlize)
    ax.set_aspect('equal',adjustable='datalim')
    fig.colorbar(surf,ax=ax)
    plt.savefig('picture/'+name+'.png')
def mag_gen(i):
    model_uniform=model({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    x=model_uniform.to_centroid(model_uniform.trajectory_l).real
    y=model_uniform.to_centroid(model_uniform.trajectory_l).imag
    uniform_mag=model_uniform.get_magnifaction2(tol)
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.Tol=VBBLtol
    VBBL.BinaryLightCurve
    y1 = -b_map[i]*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = b_map[i]*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), b_map[i], alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return i
    #return (x,y,VBBL_mag,uniform_mag,i)
if __name__=="__main__":
    start = time.monotonic()
    trajectory_n=1000
    sample_n=1200
    tol=1e-3
    VBBLtol=tol*1e-4
    t_0=2452848.06;t_E=61.5
    times=np.linspace(t_0-0.*t_E,t_0+2.0*t_E,trajectory_n);alphadeg=90
    tau=(times-t_0)/t_E
    alpha_VBBL=np.pi+alphadeg/180*np.pi
    q=1e-9;s=4;rho=1e-3
    b_map=np.linspace(-4.0,3.0,sample_n)
    VBBL_mag_map=np.zeros((sample_n,trajectory_n))
    mycode_map=np.zeros((sample_n,trajectory_n))
    traj_x=np.zeros((sample_n,trajectory_n))
    traj_y=np.zeros((sample_n,trajectory_n))
    result=[]
    for i in range(1):
        '''q=10**(np.random.uniform(-9.,0.))
        s=np.random.uniform(0.1,4.)
        rho=10**(np.random.uniform(-3.,-1.))'''
        q=0.0006943022058639565;s=3.6261560264322408;rho=0.010084752886582907
        print(f'parameter:q={q};s={s};rho={rho}')
        with Pool(processes=150) as pool:
            i= pool.imap(mag_gen,range(sample_n))
            while True:
                try:
                    result.append(next(i))
                except StopIteration:
                    break
                except Exception as e:
                    # do something
                    result.append(e)
        print(f'time took: {time.monotonic() - start:.1f}')
        print(result)
        '''for x,y,vbbl,uniform,i in pool.imap(mag_gen,range(sample_n)):
                VBBL_mag_map[i,:]=vbbl
                traj_x[i,:]=x
                traj_y[i,:]=y
                mycode_map[i,:]=uniform
            pool.close()
            pool.join()
        delta=np.abs(VBBL_mag_map-mycode_map)'''
        '''draw_colormap(delta,'delta',colors.LogNorm())
        draw_colormap(np.abs(VBBL_mag_map),'VBBL2.0',colors.LogNorm())
        draw_colormap(np.abs(mycode_map),'mycode',colors.LogNorm())'''
        '''max_delta=np.max(delta)
        if max_delta>tol:
            error_number=(delta>tol).sum()
            delta_max=np.where(np.round(delta-max_delta,4)==0)[0][0]
            print('max delta=',max_delta)
            print('error point number=',error_number)
            print('max error correspond b',b_map[delta_max])'''
        '''np.savetxt('data/mag_map_x.txt',traj_x)
            np.savetxt('data/mag_map_y.txt',traj_y)
            np.savetxt('data/mag_MulensModel.txt',VBBL_mag_map)
            np.savetxt('data/mag_uniform.txt',mycode_map)'''