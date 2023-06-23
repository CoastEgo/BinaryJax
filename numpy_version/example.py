import matplotlib.pyplot as plt
from MulensModel import Model,caustics
from uniform_model_noimage import model
import VBBinaryLensing
import time
import cProfile
import timeit
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
if __name__=="__main__":
    sample_n=120
    b_map =np.linspace(-4.0,3.0,sample_n)
    b=b_map[51]
    t_0=2452848.06;t_E=61.5;alphadeg=90
    q=1e-3;s=1;rho=0.001
    tol=1e-2
    trajectory_n=1
    alpha=alphadeg*2*np.pi/360
    times=np.linspace(t_0-0.*t_E,t_0+1.5*t_E,trajectory_n)
    ####################
    start=time.perf_counter()
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    uniform_mag=model_uniform.get_magnifaction2(tol,retol=tol)
    end=time.perf_counter()
    print(uniform_mag)
    time_optimal=end-start
    '''profiler = cProfile.Profile()
    profiler.run('model_uniform.get_magnifaction2(tol, retol=tol)')
    # 显示前10个函数调用的统计信息
    profiler.print_stats(sort='cumtime')'''
    print('low_mag_mycode=',time_optimal)
    #print(uniform_mag)
    ##################
    '''
    start=time.perf_counter()
    model_1S2L = Model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha': alphadeg})
    model_1S2L.set_magnification_methods([t_0-1.6*t_E, 'VBBL',t_0+1.5*t_E])
    model_1S2L.set_magnification_methods_parameters({'VBBL': {'accuracy': tol}})
    Mulens_mag=model_1S2L.get_magnification(time=times)
    end=time.perf_counter()
    time_Mulens=end-start'''
    #############
    start=time.perf_counter()
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol=tol
    VBBL.BinaryLightCurve
    tau=(times-t_0)/t_E
    alpha+=np.pi
    y1 = -b*np.sin(alpha) + tau*np.cos(alpha)
    y2 = b*np.cos(alpha) + tau*np.sin(alpha)
    params = [np.log(s), np.log(q), b, alpha, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    end=time.perf_counter()
    VBBLtimes=end-start
    #print('Mulens',time_Mulens)
    print('VBBLtimes',VBBLtimes)
    #print('slowthanMulens=',time_optimal/time_Mulens)
    print('slowthanVBBL=',time_optimal/VBBLtimes)
    bool2=1
    if 0:
        '''theta=np.linspace(0,2*np.pi,100)
        trajectory_l=model_uniform.trajectory_l
        zeta=model_uniform.to_centroid(model_uniform.get_zeta_l(trajectory_l[204],theta))
        fig=plt.figure(figsize=(6,6))
        plt.plot(zeta.real,zeta.imag,color='r',linewidth=0.015)
        caustic_1=caustics.Caustics(q,s)
        caustic_1.plot(5000,s=0.005)
        x,y=caustic_1.get_caustics()
        x=caustic_1.critical_curve.x
        y=caustic_1.critical_curve.y
        plt.scatter(x,y,s=0.005)'''
        curve=model_uniform.image_contour_all[204]
        for k in range(len(curve)):
            cur=model_uniform.to_centroid(curve[k])
            linewidth=0.15
            plt.figure()
            plt.plot(cur.real,cur.imag,linewidth=linewidth)
            plt.axis('equal')
            plt.savefig('picture/204_'+str(k)+'.pdf',format='pdf')
    if 0:
        fig=plt.figure(figsize=(6,6))
        #axis=plt.axes()
        axis=plt.axes(xlim=(-0.005,0.005),ylim=(-0.005,0.005))
        #model_1S2L.plot_trajectory()
        caustic_1=caustics.Caustics(q,s)
        caustic_1.plot(5000,s=0.5)
        x,y=caustic_1.get_caustics()
        x=caustic_1.critical_curve.x
        y=caustic_1.critical_curve.y
        axis.scatter(x,y,s=0.05)
        model_uniform.draw_anim(fig,axis)
    if bool2:
        plt.figure('mag')
        plt.plot(tau,uniform_mag,label='optimal_mycode')
        #plt.plot(times[203:205],Mulens_mag[203:205],label='Mulens')
        plt.plot(tau,VBBL_mag,label='VBBL')
        plt.legend()
        plt.savefig('picture/mag.png')
    if bool2:
        plt.figure('de-mag')
        plt.plot(tau,np.abs((VBBL_mag-uniform_mag)),label='$\Delta$_optimal')
        print(np.max(np.abs((VBBL_mag-uniform_mag))/np.abs(VBBL_mag)))
        print(np.argmax(np.abs((VBBL_mag-uniform_mag))))
        plt.yscale('log')
        plt.legend()
        plt.savefig('picture/delta_mag.png')