import numpy as np
from uniform_model_numpy import model
import VBBinaryLensing
from multiprocessing import Pool
import multiprocessing as mp
import sys
import time
import traceback
sys.setrecursionlimit(10000)
np.seterr(divide='ignore', invalid='ignore')
def mag_gen(i):
    model_uniform=model({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    x=model_uniform.to_centroid(model_uniform.trajectory_l).real
    y=model_uniform.to_centroid(model_uniform.trajectory_l).imag
    uniform_mag=model_uniform.get_magnifaction(tol)
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.Tol=tol
    VBBL.RelTol=tol/10
    VBBL.BinaryLightCurve
    y1 = -b_map[i]*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = b_map[i]*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), b_map[i], alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    #return i
    return (x,y,VBBL_mag,uniform_mag,i)
if __name__=="__main__":
    start = time.monotonic()
    trajectory_n=400
    sample_n=1200
    tol=1e-3
    t_0=2452848.06;t_E=61.5
    times=np.linspace(t_0-0.02*t_E,t_0+0.02*t_E,trajectory_n);alphadeg=90
    tau=(times-t_0)/t_E
    alpha_VBBL=np.pi+alphadeg/180*np.pi
    q=1e-5;s=1.0;rho=1e-3
    b_map=np.linspace(-0.08,0.04,sample_n)
    VBBL_mag_map=np.zeros((sample_n,trajectory_n))
    uni_mag_map=np.zeros((sample_n,trajectory_n))
    traj_x=np.zeros((sample_n,trajectory_n))
    traj_y=np.zeros((sample_n,trajectory_n))
    result=[]
    with Pool(processes=100) as pool:
        '''i= pool.imap(mag_gen,range(sample_n))
        while True:
            try:
                result.append(next(i))
            except StopIteration:
                break
            except Exception as e:
                # do something
                result.append(e)
    print(f'time took: {time.monotonic() - start:.1f}')
    print(result)'''
        for x,y,vbbl,uniform,i in pool.imap(mag_gen,range(sample_n)):
            VBBL_mag_map[i,:]=vbbl
            traj_x[i,:]=x
            traj_y[i,:]=y
            uni_mag_map[i,:]=uniform
        np.savetxt('data/mag_map_x.txt',traj_x)
        np.savetxt('data/mag_map_y.txt',traj_y)
        np.savetxt('data/mag_MulensModel.txt',VBBL_mag_map)
        np.savetxt('data/mag_uniform.txt',uni_mag_map)
        pool.close()
        pool.join()