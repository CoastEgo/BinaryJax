import numpy as np
from uniform_model_numpy import model
from MulensModel import Model,caustics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
np.seterr(divide='ignore', invalid='ignore')
b_map=np.linspace(-1,1,120)
trax=np.zeros((120,400))
tray=np.zeros((120,400))
test=np.zeros((120,400))
t_0=2452848.06;t_E=61.5
q=1;alphadeg=90;s=1.0;rho=1e-3
tol=1e-2
trajectory_n=400
alpha=alphadeg*2*np.pi/360
for i in range(120):
    b=b_map[i]
    times=np.linspace(t_0-1.0*t_E,t_0+1.0*t_E,trajectory_n)
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    test[i]=model_uniform.get_magnifaction2(tol)
    trax[i]=model_uniform.to_centroid(model_uniform.trajectory_l).real
    tray[i]=model_uniform.to_centroid(model_uniform.trajectory_l).imag
fig,ax=plt.subplots(figsize=(6,6))
surf=ax.pcolormesh(trax,tray,test,cmap=cm.Greys)
caustic_1=caustics.Caustics(q,s)
x,y=caustic_1.get_caustics()
ax.scatter(x,y,s=0.05)
ax.set_aspect('equal')
fig.colorbar(surf,ax=ax)
plt.savefig('picture/test2.png')
