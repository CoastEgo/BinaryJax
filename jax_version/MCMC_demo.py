import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from uniform_model_jax import model
import VBBinaryLensing
# 生成数据
np.random.seed(0)
b=0.2
t_0=2452848.06;t_E=61.5;alphadeg=90
q=0.04;s=1.3;rho=0.009
tol=1e-2
trajectory_n=50
VBBL = VBBinaryLensing.VBBinaryLensing()
alpha_VBBL=np.pi+alphadeg/180*np.pi
VBBL.RelTol=tol
VBBL.BinaryLightCurve
times=np.linspace(t_0-2.*t_E,t_0+2.*t_E,trajectory_n)
tau=(times-t_0)/t_E
y1 = -b*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
y2 = b*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
params = [np.log(s), np.log(q), b, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
VBBL_mag=np.array(VBBL_mag)
VBBL_mag*=(1+ np.random.normal(0, 0.001, size=VBBL_mag.shape))
##
alpha=alphadeg*2*jnp.pi/360
times=jnp.linspace(t_0-2.*t_E,t_0+2.*t_E,trajectory_n)
mean=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                    'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
print('jit complete')
def model_MCMC(times, mag):
    t_0=2452848.06;t_E=61.5;b=0.2
    alphadeg=90;tol=1e-2
    #q=0.04;s=1.3
    q_range = numpyro.sample('q_range', dist.Uniform(-3, 0.))
    q= numpyro.deterministic('q',10**q_range)
    s = numpyro.sample('s', dist.Uniform(0.8, 1.3))
    rho_range= numpyro.sample('rho_range',dist.Uniform(-3, 0.))
    rho= numpyro.deterministic('rho',10**rho_range)
    mean=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('obs', dist.Normal(mean, sigma), obs=mag)
init_strategy=numpyro.infer.init_to_value(values={'q_range':-1.,'s':1.0,'rho_range':-1.})#'q_range':0.,'s':1.0,
# 运行 MCMC 推断
nuts_kernel = NUTS(model_MCMC,init_strategy=init_strategy,forward_mode_differentiation=True)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500,jit_model_args=True)
mcmc.run(jax.random.PRNGKey(0), times=times, mag=VBBL_mag)
posterior_samples = mcmc.get_samples()

# 获取后验分布的均值和标准差
q_mean = np.mean(posterior_samples['q'])
q_std = np.std(posterior_samples['q'])
s_mean = np.mean(posterior_samples['s'])
s_std = np.std(posterior_samples['s'])
rho_mean = np.mean(posterior_samples['rho'])
rho_std = np.std(posterior_samples['rho'])

# 打印参数的后验分布均值和标准差
print(f"q: {q_mean} +/- {q_std}")
print(f"s: {s_mean} +/- {s_std}")
print(f"rho: {rho_mean} +/- {rho_std}")
