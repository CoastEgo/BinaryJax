import jax.numpy as jnp
import numpy as np
from uniform_model_jax import model
from jax import jacfwd
import matplotlib.pyplot as plt
from matplotlib import gridspec
#plt.rcParams['text.usetex'] = True
def uniform(delta_time):
    t_0=2452848.06;t_E=61.5;alphadeg=90
    trajectory_n=100
    times=jnp.linspace(t_0-0.75*t_E,t_0+0.75*t_E,trajectory_n)+delta_time
    q=0.001;s=1;rho=0.001
    tol=1e-2
    b=0.2
    model_uniform=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times})
    uniform_mag=model_uniform.get_magnifaction2(tol)
    return uniform_mag
jacobian_fn = jacfwd(uniform)  # it returns the function in charge of computing jacobian
mag_grad = jacobian_fn(0.)
print(mag_grad)
'''
x=np.linspace(t_0-0.75*t_E,t_0+0.75*t_E,trajectory_n)
fig=plt.figure()
gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
ax0=plt.subplot(gs[0])
line0,=ax0.plot(x,mag,'k-')
ax1=plt.subplot(gs[1],sharex=ax0)
line1,=ax1.plot(x,mag_grad,'k--')
ax0.grid(True)
ax1.grid(True)
plt.setp(ax0.get_xticklabels(),visible=False)
yticks=ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax0.legend((line0,line1),('A',r'$\frac{\partial A}{\partial t_0}$'))
plt.subplots_adjust(hspace=.0)
plt.savefig('picture/grad.png')'''