import numpy as np
import matplotlib.pyplot as plt
import corner
## trace plot
def plot(name_unique,num_chains,method):
    if method=='mcmc':
        samples=np.loadtxt('/home/coast/Documents/astronomy/microlensing/microlensing/dataFit/chain_mcmc_%s.txt'%name_unique)
        name=['t_0','u_0','t_E','rho','alpha_deg','s','q','chis2']
        print(np.min(samples[:,-1]))
        ## corner plot
        fig=corner.corner(samples,labels=name,quantiles=[0.16,0.5,0.84],show_titles=True)
        fig.savefig('cornerplot_mcmc_%s.png'%name_unique)
        plt.figure(figsize=(10,10))
        for i in range(len(name)):
            plt.subplot(4,2,i+1)
            for j in range(num_chains):
                plt.plot(samples[j::num_chains,i])
            plt.ylabel(name[i])
            plt.xlabel('step')
        plt.tight_layout()
        plt.savefig('traceplot_mcmc_%s.png'%name_unique)
    elif method=='hmc':
        samples=np.loadtxt('/home/coast/Documents/astronomy/microlensing/microlensing/dataFit/posterior_samples_%s.txt'%name_unique)
        name=['t_0','u_0','t_E','alpha_deg','logs','logq','chis2']
        print(np.min(samples[:,-1]))
        exit()
        ## corner plot
        fig=corner.corner(samples,labels=name,quantiles=[0.16,0.5,0.84],show_titles=True)
        fig.savefig('cornerplot_hmc_%s.png'%name_unique)
        length=samples.shape[0]/num_chains
        length=int(length)
        plt.figure(figsize=(10,10))
        for i in range(len(name)):
            plt.subplot(4,2,i+1)
            for j in range(num_chains):
                plt.plot(samples[j*length:(j+1)*length,i],label='chain %d'%j)
            plt.ylabel(name[i])
            plt.xlabel('step')
        plt.tight_layout()
        plt.legend()
        plt.savefig('traceplot_hmc_%s.png'%name_unique)
    else:
        raise ValueError('method should be mcmc or hmc')
if __name__=='__main__':
    name='reparam_all'
    plot(name,14,'hmc')