import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MulensModel import Model,caustics
from uniform_model_numpy import model
from function_numpy import search,search_first_postion
import sys
import time
sys.setrecursionlimit(10000)
np.seterr(divide='ignore', invalid='ignore')
if __name__=="__main__":
    b_map=np.linspace(-0.08,0.04,120)
    b=b_map[79]
    t_0=2452848.06;t_E=61.5
    q=1e-4;alphadeg=90;s=1.0;rho=1e-3
    trajectory_n=300
    alpha=alphadeg*2*np.pi/360
    start=time.perf_counter()
    times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    uniform_mag=model_uniform.get_magnifaction()
    end=time.perf_counter()
    print('uniform=',end-start)
    start=time.perf_counter()
    model_1S2L = Model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
    model_1S2L.set_magnification_methods([t_0-1.6*t_E, 'VBBL',t_0+1.5*t_E])
    model_1S2L.set_magnification_methods_parameters({'VBBL': {'accuracy': 1e-3}})
    Mulens_mag=model_1S2L.get_magnification(time=times)
    end=time.perf_counter()
    print('mulensmodel=',end-start)
    bool2=1
    if 0:
        fig=plt.figure(figsize=(6,6))
        #axis=plt.axes()
        axis=plt.axes(xlim=(-2,2),ylim=(-2,2))
        model_1S2L.plot_trajectory()
        caustic_1=caustics.Caustics(q,s)
        caustic_1.plot(5000,s=0.5)
        x,y=caustic_1.get_caustics()
        x=caustic_1.critical_curve.x
        y=caustic_1.critical_curve.y
        axis.scatter(x,y,s=0.05)
        model_uniform.draw_anim(fig,axis)
    if bool2:
        plt.figure('mag')
        plt.plot(times,Mulens_mag,label='MulensModel')
        plt.plot(times,uniform_mag,label='uniform')
        plt.legend()
        plt.savefig('picture/magnification_optimal.png')
    if 1:
        plt.figure('de-mag')
        plt.plot(times,np.abs((Mulens_mag-uniform_mag)),label='$\Delta$_optimal')
        plt.yscale('log')
        plt.legend()
        plt.savefig('picture/delta_mag_optimal.png')