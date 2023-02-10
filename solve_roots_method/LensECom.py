import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from MulensModel import MulensData, Model, Event
def to_centroid(x):
    return -(x-delta_x)
def to_lowmass(x):
    return -x+delta_x
def verify(zeta,z):
    return  z-m1/(np.conj(z)-s)-m2/np.conj(z)-zeta
q=0.05;s=1.0;alphadeg=90;b=0.1;n=50;alpha=alphadeg*2*np.pi/360
delta_x=s/(1+q)
m1=1/(1+q);m2=q/(1+q);posi=np.array([0,0])##m1较大 m2较小
pos_x_l=np.array([s,0])
pos_y_l=np.array([0,0])
zeta_l=np.array([to_lowmass(i*np.cos(alpha)-b*np.sin(alpha))+1j*(b*np.cos(alpha)+i*np.sin(alpha)) for i in np.linspace(-2,2,n)])
zeta_conj=np.conj(zeta_l)
c0=s**2*zeta_l*m2**2
c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
c2=zeta_l-s**2*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
c5=(s-zeta_conj)*zeta_conj
fig=plt.figure(figsize=(6,6))
axis=plt.axes(xlim=(-2,2),ylim=(-2,2))
model_1S2L = Model({'t_0': 2452848.06, 'u_0': b, 't_E': 61.5,
                    'rho': 0.00096, 'q': q, 's': s, 'alpha': alphadeg})
model_1S2L.set_magnification_methods([2452833., 'VBBL', 2452845.])
model_1S2L.plot_trajectory(caustics=True)
theta = np.linspace(0, 2 * np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
axis.plot(x,y)
axis.scatter(-(pos_x_l-delta_x),pos_y_l,color='green')
ims=[]
for i in range(n):
    p=[c5[i],c4[i],c3[i],c2[i],c1[i],c0[i]]
    roots=np.roots(p)
    cond=np.round(verify(zeta_l[i],roots),4)==0
    roots=roots[cond]
    image_pos_x=np.round(to_centroid(roots.real),4)
    image_pos_y=np.round(roots.imag,4)
    img1,=axis.scatter(image_pos_x,image_pos_y,color='b').findobj()
    img2,=axis.scatter(to_centroid(zeta_l[i].real),zeta_l[i].imag,color='r').findobj()
    ims.append([img1,img2])
ani=animation.ArtistAnimation(fig,ims,interval=100,repeat_delay=1000)
ani.save('animation_complex.gif',writer='pillow')