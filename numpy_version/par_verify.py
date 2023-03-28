import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from MulensModel import Model,caustics
from uniform_model_numpy import model
from mpl_toolkits.mplot3d import Axes3D
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
    model_1S2L = Model({'t_0': t_0, 'u_0': b_map[i], 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
    model_1S2L.set_magnification_methods([t_0-1.6*t_E, 'VBBL',t_0+1.5*t_E])
    model_1S2L.set_magnification_methods_parameters({'VBBL': {'accuracy': tol}})
    Mulens_mag=model_1S2L.get_magnification(time=times)
    Mulens_traj_class=model_1S2L.get_trajectory(times)
    uniform_mag=model_uniform.get_magnifaction(tol)
    uni_mag_map[i,:]=uniform_mag
    #return i
    return (Mulens_traj_class.x,Mulens_traj_class.y,Mulens_mag,uniform_mag,i)
if __name__=="__main__":
    start = time.monotonic()
    trajectory_n=300
    sample_n=1200
    tol=1e-3
    t_0=2452848.06;t_E=61.5
    times=np.linspace(t_0-0.015*t_E,t_0+0.015*t_E,trajectory_n);alphadeg=90
    q=1e-4;s=1.0;rho=1e-3
    b_map=np.linspace(-0.08,0.04,sample_n)
    Mul_mag_map=np.zeros((sample_n,trajectory_n))
    uni_mag_map=np.zeros((sample_n,trajectory_n))
    Mul_traj_x=np.zeros((sample_n,trajectory_n))
    Mul_traj_y=np.zeros((sample_n,trajectory_n))
    result=[]
    with Pool(processes=192) as pool:
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
        for x,y,Mulens,uniform,i in pool.imap(mag_gen,range(sample_n)):
            Mul_mag_map[i,:]=Mulens
            Mul_traj_x[i,:]=x
            Mul_traj_y[i,:]=y
            uni_mag_map[i,:]=uniform
        np.savetxt('data/mag_map_x.txt',Mul_traj_x)
        np.savetxt('data/mag_map_y.txt',Mul_traj_y)
        np.savetxt('data/mag_MulensModel.txt',Mul_mag_map)
        np.savetxt('data/mag_uniform.txt',uni_mag_map)
        pool.close()
        pool.join()