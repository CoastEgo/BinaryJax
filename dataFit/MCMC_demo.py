import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from ..binaryJax import model
import VBBinaryLensing
numpyro.enable_x64()
# 生成数据
np.random.seed(0)
b=0.2;t_0=2452848.06;t_E=61.5;alphadeg=90;q=0.04;s=1.3;rho=0.009
tol=1e-2
trajectory_n=500
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
def model_MCMC(times, mag):
    t_0=2452848.06
    tol=0.01
    ###参数采样
    alphadeg_range=numpyro.sample('alphadeg_range', dist.Uniform(0., 1))
    t_E_range=numpyro.sample('t_E_range', dist.Uniform(0., 1))
    q_range = numpyro.sample('q_range', dist.Uniform(-1.4, -1.25))
    rho_range= numpyro.sample('rho_range',dist.Uniform(-2.3, -2.0))
    #参数计算
    alphadeg=numpyro.deterministic('alphadeg',alphadeg_range*360)
    b=numpyro.sample('b', dist.Uniform(0,2))
    q= numpyro.deterministic('q',10**q_range)
    s = numpyro.sample('s', dist.Uniform(1.28,1.31))
    t_E=numpyro.deterministic('t_E',100*t_E_range)
    rho= numpyro.deterministic('rho',10**rho_range)
    mean=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    numpyro.sample('obs', dist.Normal(mean, tol*mean), obs=mag)
init_strategy=numpyro.infer.init_to_value(values={'alphadeg_range':0.5,'t_E_range':0.5,'b':1.,'q_range':-1.26,'s':1.281,'rho_range':-2.2})#'q_range':0.,'s':1.0,
# 运行 MCMC 推断
nuts_kernel = NUTS(model_MCMC,step_size=0.001,adapt_step_size=False,init_strategy=init_strategy,forward_mode_differentiation=True)
#,target_accept_prob=0.9
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500,jit_model_args=True)
mcmc.run(jax.random.PRNGKey(0), times=times, mag=VBBL_mag)
posterior_samples = mcmc.get_samples()

# 获取后验分布的均值和标准差
mcmc.print_summary(exclude_deterministic=False)
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
