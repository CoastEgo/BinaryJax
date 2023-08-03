import sys
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=30'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
import matplotlib.pyplot as plt
import corner
from binaryJax import model
import VBBinaryLensing
# 生成数据
np.random.seed(0)##-1.3979，#-2.04
b=0.2;t_0=2452848.06;t_E=61.5;alphadeg=90;q=0.04;s=1.28;rho=0.009
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
    tol=0.001
    ###参数采样
    alphadeg_range=numpyro.sample('alphadeg_range', dist.Uniform(0., 1))
    t_E_range=numpyro.sample('t_E_range', dist.Uniform(0., 1))
    q_range = numpyro.sample('q_range', dist.Uniform(-1.45, -1.15))
    rho_range= numpyro.sample('rho_range',dist.Uniform(-2.1, -1.95))
    #参数计算
    alphadeg=numpyro.deterministic('alphadeg',alphadeg_range*360)
    b=numpyro.sample('b', dist.Uniform(0,2))
    q= numpyro.deterministic('q',10**q_range)
    s = numpyro.sample('s', dist.Uniform(1.27,1.29))
    t_E=numpyro.deterministic('t_E',100*t_E_range)
    rho= numpyro.deterministic('rho',10**rho_range)
    mean=model({'t_0': t_0, 'u_0': b, 't_E': t_E,
                        'rho': rho, 'q': q, 's': s, 'alpha_deg': alphadeg,'times':times,'retol':tol})
    numpyro.sample('obs', dist.Normal(mean, tol*mean), obs=mag)
#init_strategy=numpyro.infer.init_to_value(values={'alphadeg_range':0.5,'t_E_range':0.5,'b':1.,'q_range':-1.26,'s':1.2,'rho_range':-2.2})
# 运行 MCMC 推断
nuts_kernel = NUTS(model_MCMC,step_size=0.001,max_tree_depth=7,forward_mode_differentiation=True)
#,target_accept_prob=0.9
mcmc = MCMC(nuts_kernel, num_samples=400, num_warmup=500,jit_model_args=True,num_chains=30)
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
t_E_mean = np.mean(posterior_samples['t_E'])
t_E_std = np.std(posterior_samples['t_E'])
alphadeg_mean = np.mean(posterior_samples['alphadeg'])
alphadeg_std = np.std(posterior_samples['alphadeg'])
b_mean = np.mean(posterior_samples['b'])
b_std = np.std(posterior_samples['b'])
# 打印参数的后验分布均值和标准差
print(f"q: {q_mean} +/- {q_std}")
print(f"s: {s_mean} +/- {s_std}")
print(f"rho: {rho_mean} +/- {rho_std}")
print(f"b: {b_mean} +/- {b_std}")
print(f"t_E: {t_E_mean} +/- {t_E_std}")
print(f"alphadeg: {alphadeg_mean} +/- {alphadeg_std}")
###绘制corner图
samples = np.column_stack([posterior_samples["q"], posterior_samples["s"],posterior_samples["rho"],posterior_samples["t_E"],posterior_samples["b"],posterior_samples["alphadeg"]])
figure = corner.corner(samples, labels=["q", "s","rho",'t_E',"b","alphadeg"], quantiles=[0.025, 0.5, 0.975], show_titles=True)
figure.savefig('corner')
###绘制拟合与数据图
fitdata=model({'t_0': t_0, 'u_0': b_mean, 't_E': t_E_mean,
                        'rho': rho_mean, 'q': q_mean, 's': s_mean, 'alpha_deg': alphadeg_mean,'times':times,'retol':0.001})
plt.figure()
plt.plot(times,fitdata,c='r')
plt.scatter(times,VBBL_mag,s=15,color='grey')
plt.savefig('MCMCfit&mock_data')
