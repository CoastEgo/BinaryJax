import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from MulensModel import Model,caustics
from uniform_model_numpy import model
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.setrecursionlimit(10000)
np.seterr(divide='ignore', invalid='ignore')
if __name__=="__main__":
    trajectory_n=30
    sample_n=20
    b_map=np.linspace(-1,1,sample_n)
    Mul_mag_map=np.array([])
    uni_mag_map=np.array([])
    Mul_traj_x=np.array([])
    Mul_traj_y=np.array([])
    for i in range(sample_n):
        b=b_map[i]
        t_0=2452848.06;t_E=61.5;q=0.05;alphadeg=30;s=0.8;rho=0.05;alpha=alphadeg*2*np.pi/360;delta_x=s/(1+q);m1=1/(1+q);m2=q/(1+q);posi=np.array([0,0])##m1较大 m2较小
        times=np.linspace(t_0-1.5*t_E,t_0+1.5*t_E,trajectory_n)
        model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
        model_1S2L = Model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
        model_1S2L.set_magnification_methods([t_0-1.5*t_E, 'VBBL',t_0+1.5*t_E])
        Mulens_mag=model_1S2L.get_magnification(time=times)
        Mulens_traj_class=model_1S2L.get_trajectory(times)
        Mul_traj_x=np.append(Mul_traj_x,Mulens_traj_class.x)
        Mul_traj_y=np.append(Mul_traj_y,Mulens_traj_class.y)
        Mul_mag_map=np.append(Mul_mag_map,Mulens_mag)
        uniform_mag=model_uniform.get_magnifaction()
        uni_mag_map=np.append(uni_mag_map,uniform_mag)
        if i==10:
            print(10)
        delta=np.abs(uni_mag_map-Mul_mag_map)/Mul_mag_map
        uniform_trajectory=model_uniform.to_centroid(model_uniform.trajectory_l)
    if 1:
        fig=plt.figure(figsize=(6,6))
        #caustic=caustics.Caustics(q,s)
        #caustic.plot(5000,s=0.5)
        ax = fig.add_subplot(projection='3d')
        #ax.scatter(Mul_traj_x,Mul_traj_y,uni_mag_map,color='b',s=0.1)
        ax.scatter(Mul_traj_x,Mul_traj_y,np.abs(Mul_mag_map-uni_mag_map)/Mul_mag_map,color='r',s=0.1)
        delta=np.abs(Mul_mag_map-uni_mag_map)/Mul_mag_map
        max_delta=np.max(delta)
        print(np.where(delta==max_delta)[0])
        #surf=axis.plot_trisurf(uni_traj_x,uni_traj_y,uni_mag_map,cmap=cm.jet,linewidth=0.1,antialiased=False)
        #suf_mu=axis.plot_trisurf(Mul_traj_x,Mul_traj_y,Mul_mag_map,cmap=cm.jet,linewidth=0.1)
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        if 0:
            fig=plt.figure(figsize=(6,6))
            axis=plt.axes(xlim=(-2,2),ylim=(-2,2))
            model_1S2L.plot_trajectory()
            caustic_1=caustics.Caustics(q,s)
            caustic_1.plot(5000,s=0.5)
            x,y=caustic_1.get_caustics()
            x=caustic_1.critical_curve.x
            y=caustic_1.critical_curve.y
            axis.scatter(x,y,s=0.05)
            model_uniform.draw_anim(fig,axis)
        if 0:
            plt.figure('mag')
            plt.plot(times,Mulens_mag,label='MulensModel')
            plt.plot(times,uniform_mag,label='uniform')
            plt.legend()
            plt.savefig('picture/magnification.png')
        if 0:
            plt.figure('de-mag')
            plt.plot(times,np.abs(Mulens_mag-uniform_mag)/Mulens_mag,label='$\Delta$')
            delta=np.abs(Mulens_mag-uniform_mag)/Mulens_mag
            plt.legend()
            plt.savefig('picture/delta_magnification.png')