import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import matplotlib.animation as animation
from MulensModel import MulensData, Model, Event
def initial(s,q):
    global ym
    ym=np.zeros([2,2])
    ym[0]=[round(s/(1+q),4),0.0]
    ym[1]=[round(-s*q/(1+q),4),0.0]
    global epsilon
    epsilon=np.zeros(2)
    epsilon[0]=q/(1+q)
    epsilon[1]=1/(1+q)
def leq(y):
    temp=np.zeros(2)
    for i in range(len(ym)):
        temp+=(y-ym[i])*(np.linalg.norm(y-ym[1-i])**2)*epsilon[i]
    return (y-u)*(np.linalg.norm(y-ym[0])**2)*(np.linalg.norm(y-ym[1])**2)-temp
def solvescip():
    temp=1
    for i in range(100):
        solution = root(fun=leq,x0=initial_guess[i])
        if solution.success:
            if temp==1:
                rel=np.zeros([1,2])
                rel[0,:]=np.round(solution.x,4)
                temp=0
                continue
            rel=np.append(rel,np.round(solution.x,4).reshape(1,2),axis=0)
    result=np.unique(rel,axis=0)
    return result[:,0],result[:,1]
def initi_gues():
    x=np.linspace(-2,2,10).reshape([10,1])
    y=np.linspace(-2,2,10).reshape([1,10])
    xx,yy=np.meshgrid(x,y)
    return np.array((xx.ravel(),yy.ravel())).T
if 1:
    q=0.05;s=1.0;alphadeg=90;n=50;b=0.1
    initial(s,q)
    initial_guess=initi_gues()
    alpha=alphadeg*2*np.pi/360
    pos=np.array([[i*np.cos(alpha)-b*np.sin(alpha),b*np.cos(alpha)+i*np.sin(alpha)] for i in np.linspace(-2,2,n)])
    fig=plt.figure(figsize=(6,6))
    model_1S2L = Model({'t_0': 2452848.06, 'u_0': b, 't_E': 61.5,
                    'rho': 0.00096, 'q': q, 's': s, 'alpha': alphadeg})
    model_1S2L.set_magnification_methods([2452833., 'VBBL', 2452845.])
    model_1S2L.plot_trajectory(caustics=True)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.xticks(np.arange(-2,2,0.5))
    plt.yticks(np.arange(-2,2,0.5))
    plt.plot(x,y)
    plt.plot(pos[:,0],pos[:,1])
    plt.scatter(ym[:,0],ym[:,1],color='green')
    ims=[]
    for i in range(n):
        u=pos[i,:]
        x,y=solvescip()
        cond=(((x!=ym[0,0])|(y!=ym[0,1])) & ((x!=ym[1,0])|(y!=ym[1,1])))
        x=x[cond]
        y=y[cond]
        print(x)
        print(y)
        im1,=plt.scatter(x,y,color='b').findobj()
        im2,=plt.scatter(u[0],u[1],color='r').findobj()
        ims.append([im1,im2])
    ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=1000)
    ani.save('animation.gif',writer='pillow')
