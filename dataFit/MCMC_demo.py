import sys
import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
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
u_0=0.05;t_0=2452848.06;t_E=61.5;alpha_deg=90;q=0.04;s=1.28;rho=0.009
tol=1e-3
trajectory_n=1000
VBBL = VBBinaryLensing.VBBinaryLensing()
alpha_VBBL=np.pi+alpha_deg/180*np.pi
VBBL.RelTol=tol
VBBL.BinaryLightCurve
times=np.linspace(t_0-1.*t_E,t_0+1.*t_E,trajectory_n)
tau=(times-t_0)/t_E
y1 = -u_0*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
y2 = u_0*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
params = [np.log(s), np.log(q), u_0, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
VBBL_mag=np.array(VBBL_mag)
VBBL_mag*=(1+ np.random.normal(0, 0.001, size=VBBL_mag.shape))
##
alpha=alpha_deg*2*jnp.pi/360
times=jnp.linspace(t_0-1.*t_E,t_0+1.*t_E,trajectory_n)
parmfitname=['alpha_deg','t_E','rho','u_0','q','s']
fitlimits=[[86.4, 100],[45., 65.],[-2.1, -1.5],[0.02,0.08],[-1.42, -1.35],[1.25,1.29]]
parmfix={'t_0':2452848.06,'retol':1e-2,'times':times}
def model_MCMC(mag):
    parmfitname=['alpha_deg','t_E','rho','u_0','q','s']
    fitlimits=[[86.4, 100],[45., 65.],[-2.1, -1.5],[0.02,0.08],[-1.42, -1.35],[1.25,1.29]]
    parmfix={'t_0':2452848.06,'retol':1e-2,'times':times}
    parmdeter=[]
    for i in range(len(parmfitname)):
        temp=numpyro.sample('scaled_'+parmfitname[i],dist.Uniform(0,1))
        parmdeter+=[numpyro.deterministic(parmfitname[i],temp*(fitlimits[i][1]-fitlimits[i][0])+fitlimits[i][0])]
    parmfit=dict(zip(parmfitname,parmdeter))
    parmfit['rho']=10**parmfit['rho']
    parmfit['q']=10**parmfit['q']
    parmfit.update(parmfix)
    mean=model(parmfit)
    numpyro.sample('obs', dist.Normal(mean, parmfix['retol']*mean), obs=mag)
initvalue=[95,50,-1.8,0.05,-1.36,1.26]
for i in range(len(initvalue)):
    initvalue[i]=(initvalue[i]-fitlimits[i][0])/(fitlimits[i][1]-fitlimits[i][0])
initparmname=['scaled_'+i for i in parmfitname]
init_strategy=numpyro.infer.init_to_value(values=dict(zip(initparmname,initvalue)))
# 运行 MCMC 推断
nuts_kernel = NUTS(model_MCMC,step_size=1e-4,forward_mode_differentiation=True,max_tree_depth=8,init_strategy=init_strategy,adapt_step_size=False)
#mcmc = MCMC(nuts_kernel, num_samples=400, num_warmup=500,jit_model_args=True,num_chains=30,progress_bar=False)#
mcmc = MCMC(nuts_kernel, num_samples=100, num_warmup=50,num_chains=1)
mcmc.run(jax.random.PRNGKey(0),VBBL_mag)#parmfitname,fitlimits,parmfix
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
alphadeg_mean = np.mean(posterior_samples['alpha_deg'])
alphadeg_std = np.std(posterior_samples['alpha_deg'])
b_mean = np.mean(posterior_samples['u_0'])
b_std = np.std(posterior_samples['u_0'])
# 打印参数的后验分布均值和标准差
print(f"q: {q_mean} +/- {q_std}")
print(f"s: {s_mean} +/- {s_std}")
print(f"rho: {rho_mean} +/- {rho_std}")
print(f"u_0: {b_mean} +/- {b_std}")
print(f"t_E: {t_E_mean} +/- {t_E_std}")
print(f"alpha_deg: {alphadeg_mean} +/- {alphadeg_std}")
####trace plot
plt.figure(figsize=(12,10))
for i in range(len(parmfitname)):
    plt.subplot(3,2,i+1)
    plt.plot(posterior_samples[parmfitname[i]])
    plt.ylabel(parmfitname[i])
plt.tight_layout()
plt.savefig('traceplot')
### save sample
np.savetxt("posterior_samples.txt", 
           np.column_stack([posterior_samples['q'],
                            posterior_samples['s'],
                            posterior_samples['rho'],
                            posterior_samples['t_E'],
                            posterior_samples['u_0'],
                            posterior_samples['alpha_deg']]),
           header="q s rho t_E u_0 alpha_deg")
###绘制corner图
samples = np.column_stack([posterior_samples["q"], posterior_samples["s"],posterior_samples["rho"],posterior_samples["t_E"],posterior_samples["u_0"],posterior_samples["alpha_deg"]])
figure = corner.corner(samples, labels=["q", "s","rho",'t_E',"u_0","alpha_deg"], quantiles=[0.025, 0.5, 0.975], show_titles=True)
figure.savefig('corner_demo2')
###绘制拟合与数据图
fitdata=model({'t_0': t_0, 'u_0': b_mean, 't_E': t_E_mean,
                        'rho': 10**rho_mean, 'q': 10**q_mean, 's': s_mean, 'alpha_deg': alphadeg_mean,'times':times,'retol':0.001})
plt.figure()
plt.plot(times,fitdata,c='r')
plt.scatter(times,VBBL_mag,s=15,color='grey')
plt.savefig('MCMCfit&mock_data_demo2')
