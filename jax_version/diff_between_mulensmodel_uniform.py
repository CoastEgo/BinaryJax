import numpy as np
import matplotlib.pyplot as plt
from MulensModel import Model,caustics
from uniform_model_jax import model
import sys
sys.setrecursionlimit(10000)
np.seterr(divide='ignore', invalid='ignore')
if __name__=="__main__":
    t_0=2452848.06;t_E=61.5;q=0.05;s=1.0;alphadeg=30;b=0.1;rho=0.05;trajectory_n=100;sample_n=1000##m1较大 m2较小
    ##### parameter set#################################################
    model_MulenModel = Model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'trajectory_n':trajectory_n,'sample_n':sample_n})
    model_MulenModel.set_magnification_methods([t_0-1.5*t_E, 'VBBL',t_0+1.5*t_E])
    times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
    Mulens_mag=model_MulenModel.get_magnification(time=times)
    uniform_mag=model_uniform.mag
    if 1:
        fig=plt.figure(figsize=(6,6))
        axis=plt.axes(xlim=(-2,2),ylim=(-2,2))
        model_MulenModel.plot_trajectory()
        caustic_1=caustics.Caustics(q,s)
        caustic_1.plot(5000,s=0.5)
        x,y=caustic_1.get_caustics()
        x=caustic_1.critical_curve.x
        y=caustic_1.critical_curve.y
        plt.scatter(x,y,s=0.05)
        model_uniform.draw_anim(fig,axis)
    if 1:
        plt.figure('mag')
        plt.plot(times,Mulens_mag,label='MulensModel')
        plt.plot(times,uniform_mag,label='uniform')
        plt.legend()
        plt.savefig('picture/magnification.png')
    if 1:
        plt.figure('de-mag')
        plt.plot(times,np.abs(Mulens_mag-uniform_mag)/Mulens_mag,label='$\Delta$')
        plt.legend()
        plt.savefig('picture/delta_magnification.png')