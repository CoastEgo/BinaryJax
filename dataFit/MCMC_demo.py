import sys
import os
global numofchains
global N_pmap
N_pmap = 20
numofchains = 1
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_pmap*numofchains}'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/coast/Documents/astronomy/microlensing/jax-soft-dtw-div')
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
from functools import partial
import VBBinaryLensing
import MulensModel as mm
from MulensModel import caustics
from sdtw import sdtw
import time
if len(sys.argv)>1:
    uniquename=sys.argv[1]
else:
    uniquename='test'
def model_pmap(parms,i):
    parms['times']=parms['times'][:,i]
    return model(parms)
def get_VBBL_mag(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol):
    VBBL = VBBinaryLensing.VBBinaryLensing()
    alpha_VBBL=np.pi+alpha_deg/180*np.pi
    VBBL.RelTol=tol
    VBBL.BinaryLightCurve
    tau=(times-t_0)/t_E
    y1 = -u_0*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
    y2 = u_0*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
    params = [np.log(s), np.log(q), u_0, alpha_VBBL, np.log(rho), np.log(t_E), t_0]
    VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
    return VBBL_mag,tau
def plot_geometry(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol,unique_name):
    plt.figure(figsize=(10,10))
    ax = plt.subplot(2,1,1)
    ax.scatter(times,VBBL_mag,s=15,color='grey')
    ## plot trajectory and caustics
    ax = plt.subplot(2,1,2)
    ax.plot(-u_0*np.sin(alpha_deg/180*np.pi) + tau*np.cos(alpha_deg/180*np.pi),
            u_0*np.cos(alpha_deg/180*np.pi) + tau*np.sin(alpha_deg/180*np.pi),c='r')
    ### make x and y axis equal
    caust_model=caustics.Caustics(q,s)
    x,y=caust_model.get_caustics()
    ax.scatter(x,y,s=0.1)
    ax.axis('equal')
    #plt.savefig('trajectory&caustics_%s'%uniquename)
def get_chi2_grad(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol,VBBL_mag,error):
    def chis_grad(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol,VBBL_mag,error):
        mean=model({'t_0':t_0,'u_0':u_0,'t_E':t_E,'rho':10**rho,'alpha_deg':alpha_deg,'s':10**s,'q':10**q,'times':times,'retol':tol})
        chis=((VBBL_mag-mean)/error)**2
        return -1*jnp.sum(chis)/2
    gradient=jax.jacfwd(chis_grad,argnums=(0,1,2,3,4,5,6))(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol,VBBL_mag,error)
    return jnp.array(gradient)
def loglike_mcmc(parmfree,parmfix,data,namefit):
        parm=dict(zip(namefit,parmfree))
        parm.update(parmfix)
        parm['rho']=10**parm['rho']
        parm['q']=10**parm['q']
        parm['s']=10**parm['s']
        VBBL = VBBinaryLensing.VBBinaryLensing()
        alpha_VBBL=np.pi+parm['alpha_deg']/180*np.pi
        VBBL.RelTol=parm['retol']
        VBBL.BinaryLightCurve
        tau=(times-parm['t_0'])/parm['t_E']
        y1 = -parm['u_0']*np.sin(alpha_VBBL) + tau*np.cos(alpha_VBBL)
        y2 = parm['u_0']*np.cos(alpha_VBBL) + tau*np.sin(alpha_VBBL)
        params = [np.log(parm['s']), np.log(parm['q']), parm['u_0'], alpha_VBBL, np.log(parm['rho']), np.log(parm['t_E']), parm['t_0']]
        VBBL_mag = VBBL.BinaryLightCurve(params, times, y1, y2)
        chis=((VBBL_mag-data)/error)**2
        return -0.5*jnp.sum(chis)
def select_parm(name,value,stepsize_all,namefix,grad):
    fitlimits_all=[[value[i]-10*stepsize_all[i],value[i]+10*stepsize_all[i]] for i in range(len(value))]
    valuefix=[];valuefit=[]
    namefit=[];stepsize=[]
    gradfit=[]
    fitlimits=[]
    for i in range(len(name)):
        if name[i] in namefix:
            valuefix.append(value[i])
        else:
            gradfit.append(grad[i])
            namefit.append(name[i])
            stepsize.append(stepsize_all[i])
            fitlimits.append(fitlimits_all[i])
            valuefit.append(value[i])
    fixdict=dict(zip(namefix,valuefix))
    return fixdict,namefit,valuefit,fitlimits,stepsize,gradfit
def model_MCMC(mag,parmfitname,valuefit,fitlimits,parmfix,stepsize,error,grad,cov):
    L=jnp.linalg.cholesky(cov)
    std=jnp.sqrt(jnp.diag(cov))
    parmdeter=[]
    #parmdeter=[]
    ### gradient normalization sample at normal(value/grad,stepsize/grad) and then multiply grad
    #for i in range(len(parmfitname)):
        ## mutiply grad
        #parmdeter.append(numpyro.sample(parmfitname[i],dist.Normal(valuefit[i],stepsize[i])))
        #parmdeter.append(numpyro.sample(parmfitname[i],dist.TransformedDistribution(dist.Normal(0,stepsize[i]*grad[i]),dist.transforms.AffineTransform(valuefit[i],1/grad[i]))))
    parmdeter=numpyro.sample('param_base',dist.Normal(jnp.zeros(len(parmfitname)),jnp.ones(len(parmfitname))))
    parmdeter=(jnp.dot(L,parmdeter))/std*stepsize+jnp.array(valuefit)
    numpyro.deterministic('param',parmdeter)
    parmfit=dict(zip(parmfitname,parmdeter))
    parmfit.update(parmfix)
    parmfit['rho']=10**parmfit['rho']
    parmfit['q']=10**parmfit['q']
    parmfit['s']=10**parmfit['s']
    mean=jax.pmap(model_pmap,in_axes=(None,0))(parmfit,jnp.arange(N_pmap))
    mean=jnp.reshape(mean,(-1,),order='F')
    chis=((mag-mean)/error)**2
    numpyro.deterministic('chis',jnp.sum(chis))
    '''
    mean=jnp.reshape(mean,(-1,1),order='F')
    divergence=sdtw(jnp.reshape(mag,(-1,1)),jnp.reshape(mean,(-1,1)),jnp.reshape(error,(-1,1)),warp_penalty=1.0,temperature=0.01)
    numpyro.deterministic('dtw-divergence',divergence)
    numpyro.sample('divergence',dist.Normal(0,1),obs=divergence)'''
    with numpyro.plate('data', len(mag)):
        numpyro.sample('obs', dist.Normal(mean, error), obs=mag)
def MCMC_inferece(fixdict,namefit,valuefit,fitlimits,stepsize,error,uniquename,times,VBBL_mag):
    import emcee
    import multiprocessing
    from numpyro.diagnostics import effective_sample_size,gelman_rubin
    fixdict.update({'times':np.array(times),'retol':0.001})
    def mcmc(parmfree,parmfix,stepsize,data,namefit):
        start=time.time()
        ndim=len(parmfree)
        nwalkers=ndim*2
        pos = [parmfree + stepsize*np.random.randn(ndim) for i in range(nwalkers)]
        os.environ["OMP_NUM_THREADS"] = "1"
        with multiprocessing.Pool(nwalkers) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike_mcmc, args=(parmfix,data,namefit),pool=pool)
            ## burn-in
            pos, prob, state = sampler.run_mcmc(pos, 1000, progress=True)
            sampler.reset()
            sampler.run_mcmc(pos, 2000, progress=True)
        chain = sampler.get_chain(flat=True)
        chis2s=-2*sampler.get_log_prob(flat=True)
        accept=sampler.acceptance_fraction
        np.savetxt("chain_mcmc_%s.txt"%uniquename,np.concatenate([chain,chis2s[:,None]],axis=1),header=" ".join(namefit+['chis2']))
        np.savetxt('accept_mcmc_%s.txt'%uniquename,accept)
        parambest=chain[np.argmin(chis2s)]
        parammean=np.mean(chain,axis=0)
        paramstd=np.std(chain,axis=0)
        effective=[effective_sample_size(sampler.chain[:,:,i]) for i in range(chain.shape[1])]
        print(f'effective sample size = {effective}')
        Rcut=[gelman_rubin(sampler.chain[:,:,i]) for i in range(chain.shape[1])]
        print(f'gelman-roubin value is {Rcut}')
        for i in range(len(namefit)):
            print(f"{namefit[i]}: {parammean[i]} +/- {paramstd[i]}")
        print('time:',time.time()-start)
        print('best parameters: ')
        for i in range(len(namefit)):
            print(f"{namefit[i]}: {parambest[i]}")
        print('chis2:',np.min(chis2s))
    mcmc(valuefit,fixdict,stepsize,VBBL_mag,namefit)
def HMC_inference(fixdict,namefit,valuefit,fitlimits,stepsize,error,uniquename,times,VBBL_mag,grad,cov):
    fixdict.update({'times':jnp.reshape(times,(-1,N_pmap),order='C'),'retol':0.001})
    grad=jnp.abs(grad)
    grad=jnp.array(grad/np.min(grad))
    valuefit=jnp.array(valuefit)
    ### with reparam
    start=time.time()
    #init_strategy=numpyro.infer.init_to_value(values=dict(zip([namefit[i]+'_base' for i in range(len(namefit))],[0.]*len(namefit)))) #maybe need to used namefit_basedist
    
    ##no reparam
    #init_strategy=numpyro.infer.init_to_value(values=dict(zip(namefit,valuefit))) 

    ## normal reparam
    init_strategy=numpyro.infer.init_to_value(values={'param_base':jnp.zeros(len(namefit))})

    # 运行 MCMC 推断 
    config=dict(zip(namefit,[numpyro.infer.reparam.TransformReparam()]*len(namefit)))
    reparm_model=numpyro.handlers.reparam(model_MCMC,config=config)
    nuts_kernel = NUTS(model_MCMC,step_size=1e-2,forward_mode_differentiation=True,dense_mass=True,init_strategy=init_strategy,target_accept_prob=0.5)
    mcmc = MCMC(nuts_kernel, num_samples=500, num_warmup=1000,num_chains=numofchains,progress_bar=True)
    mcmc.run(jax.random.PRNGKey(0),jnp.array(VBBL_mag),namefit,valuefit,fitlimits,fixdict,jnp.array(stepsize),jnp.abs(jnp.array(error)),grad,cov,extra_fields=('num_steps',))
    posterior_samples = {**mcmc.get_samples(),**mcmc.get_extra_fields('num_steps')}
    end=time.time()
    print('time:',end-start)
    # 获取后验分布的均值和标准差
    mcmc.print_summary(exclude_deterministic=False)
    np.savetxt('HMC_samples_%s.txt'%uniquename,posterior_samples['param'],header=" ".join(namefit))
    figure=corner.corner(np.array(posterior_samples['param']),labels=namefit,quantiles=[0.025, 0.5, 0.975],show_titles=True)
    figure.savefig('corner_hmc_%s'%uniquename)
    '''
    ## diagnostic
    namefit.append('chis')
    sample=[posterior_samples[i] for i in namefit]
    ## save sample
    np.savetxt("posterior_samples_"+uniquename+"_numsteps.txt",posterior_samples['num_steps'].T,fmt='%d',header='num_steps')
    np.savetxt("posterior_samples_%s.txt"%uniquename,np.column_stack(sample),header=" ".join(namefit))
    # 打印参数的后验分布均值和标准差
    for i in range(len(namefit)):
        print(f"{namefit[i]}: {np.mean(sample[i])} +/- {np.std(sample[i])}")
    ####trace plot
    plt.figure(figsize=(12,10))
    for i in range(len(namefit)):
        plt.subplot(4,2,i+1)
        plt.plot(posterior_samples[namefit[i]])
        plt.ylabel(namefit[i])
    plt.tight_layout()
    plt.savefig('traceplot_%s'%uniquename)
    ###绘制corner图
    figure = corner.corner(np.column_stack(sample), labels=namefit, quantiles=[0.025, 0.5, 0.975], show_titles=True)
    figure.savefig('corner_hmc_%s'%uniquename)
    ###绘制拟合与数据图
    result=[np.mean(sample[i]) for i in range(len(namefit))]
    fitdict=dict(zip(namefit,result))
    fitdict.update(fixdict)
    fitdict['rho']=10**fitdict['rho']
    fitdict['q']=10**fitdict['q']
    fitdict['s']=10**fitdict['s']
    mean=jax.pmap(model_pmap,in_axes=(None,0))(fitdict,jnp.arange(N_pmap))
    mean=jnp.reshape(mean,(-1,),order='F')
    plt.figure()
    plt.plot(times,mean,c='r')
    plt.scatter(times,VBBL_mag,s=15,color='grey')
    plt.savefig('MCMCfit&mock_data_%s'%uniquename)'''
if __name__=='__main__':
    np.random.seed(0)
    ##no caustic is alpha_deg+=30 u_0+=0.2
    t_0=8280.094505;t_E=39.824343;alpha_deg=56.484044;u_0=0.121803;q=10**0.006738;s=10**(-0.076558);rho=10**(-2.589090)
    tol=1e-3
    trajectory_n=500
    times=np.linspace(t_0-1.*t_E,t_0+1.*t_E,trajectory_n)
    if s>1.:
        u_0+=(s-1/s)*q/(1+q)*np.sin(alpha_deg*np.pi/180)
        times-=(s-1/s)*q/(1+q)*np.cos(alpha_deg*np.pi/180)*t_E
    ###########
    ## VBBL mag
    VBBL_mag,tau=get_VBBL_mag(t_0,u_0,t_E,rho,alpha_deg,s,q,times,tol)
    ### error
    error=np.random.normal(0, 0.01, size=len(VBBL_mag))
    VBBL_mag+=error
    '''error=jnp.array(error)
    divergence=sdtw(jnp.reshape(VBBL_mag,(-1,1)),jnp.reshape(np.roll(VBBL_mag-error,2),(-1,1)),jnp.reshape(error,(-1,1)),warp_penalty=1.0,temperature=0.01)
    print(divergence)'''
    times=jnp.linspace(t_0-1.*t_E,t_0+1.*t_E,trajectory_n)
    ## calculate gradient and stepsize
    stepsize_all=[0.001,0.001,0.001,0.001,0.001,0.001,0.001]
    name=['t_0','u_0','t_E','rho','alpha_deg','s','q']
    value=[8280.094505,0.121803,39.824343,-2.589090,56.484044,-0.076558,0.006738]
    namefix=['']
    grad=get_chi2_grad(t_0,u_0,t_E,np.log10(rho),alpha_deg,np.log10(s),np.log10(q),times,tol,VBBL_mag,error)
    print('chis gradient respect to each parameter:')
    for i in range(len(name)):
        print(f'{name[i]}:  {grad[i]:.3e}')
    fisher_matrix=jax.jacfwd(get_chi2_grad,argnums=(0,1,2,3,4,5,6))(t_0,u_0,t_E,np.log10(rho),alpha_deg,np.log10(s),np.log10(q),times,tol,VBBL_mag,error)
    fisher_matrix=-1*np.array(fisher_matrix)
    np.savetxt('grad/fisher_matrix.txt',fisher_matrix)
    cov=np.linalg.inv(fisher_matrix)
    np.savetxt('grad/cov.txt',cov)
    stepsize_all=np.sqrt(np.diag(cov))
    print(' std by fisher information matrix')
    for i in range(len(name)):
        print(f'{name[i]}: {stepsize_all[i]}')
    fixdict,namefit,valuefit,fitlimits,stepsize,gradfit=select_parm(name,value,stepsize_all,namefix,grad)
    print('stepsize:',stepsize)
    HMC_inference(fixdict,namefit,valuefit,fitlimits,stepsize,error,uniquename,times,VBBL_mag,jnp.array(gradfit),jnp.array(cov))
    #MCMC_inferece(fixdict,namefit,valuefit,fitlimits,stepsize,error,uniquename,times,VBBL_mag)