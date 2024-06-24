import numpy as np
import jax.numpy as jnp
import jax
from .basic_function_jax import dot_product
@jax.jit
def basic_partial(z,theta,rho,q,s):
    delta_theta=jnp.diff(theta,axis=0)
    z_c=jnp.conj(z)
    parZetaConZ=1/(1+q)*(1/(z_c-s)**2+q/z_c**2)
    par2ConZetaZ=-2/(1+q)*(1/(z-s)**3+q/(z)**3)
    de_zeta=1j*rho*jnp.exp(1j*theta)
    detJ=1-jnp.abs(parZetaConZ)**2
    de_z=(de_zeta-parZetaConZ*jnp.conj(de_zeta))/detJ
    deXProde2X=(rho**2+jnp.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ

    # calculate the derivative of x'^x'' with respect to \theta x'^x'''
    de2_zeta = -rho*jnp.exp(1j*theta)
    de2_zetaConj = -rho*jnp.exp(-1j*theta)
    par3ConZetaZ=6/(1+q)*(1/(z-s)**4+q/(z)**4)
    de2_z = (de2_zeta-jnp.conj(par2ConZetaZ)*jnp.conj(de_z)**2-parZetaConZ*(
        de2_zetaConj-par2ConZetaZ*de_z**2))/detJ
    
    # deXProde2X_test = 1/(2*1j)*(de2_z*jnp.conj(de_z)-de_z*jnp.conj(de2_z))
    # jax.debug.print('deXProde2X_test error is {}',jnp.nansum(jnp.abs(deXProde2X_test-deXProde2X)))
    de_deXPro_de2X=1/detJ**2*jnp.imag(
        detJ*(de2_zeta*par2ConZetaZ*de_z**2+de_zeta*par3ConZetaZ*de_z**3+de_zeta*par2ConZetaZ*2*de_z*de2_z)
        +(jnp.conj(par2ConZetaZ)*jnp.conj(de_z)*jnp.conj(parZetaConZ)+parZetaConZ*par2ConZetaZ*de_z
        )*de_zeta*par2ConZetaZ*de_z**2)
    # deXProde2X = jax.lax.stop_gradient(deXProde2X)
    return deXProde2X,de_z,delta_theta,de_deXPro_de2X

@jax.jit
def error_ordinary(deXProde2X,de_z,delta_theta,z,parity,de_deXPro_de2X):
    e1=jnp.nansum(jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta**3),axis=1)
    dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta
    dAp=dAp_1*delta_theta**2*parity[0:-1]
    delta_theta_wave=jnp.abs(z[0:-1]-z[1:])**2/jnp.abs(dot_product(de_z[0:-1],de_z[1:]))
    e2=jnp.nansum(3/2*jnp.abs(dAp_1*(delta_theta_wave-delta_theta**2)),axis=1)
    e3=jnp.nansum(1/10*jnp.abs(dAp)*delta_theta**2,axis=1)
    de_dAp=1/24*(de_deXPro_de2X[0:-1]-de_deXPro_de2X[1:])*delta_theta**3*parity[0:-1]
    e4 = jnp.nansum(1/10*jnp.abs(de_dAp),axis=1) ## e4 is the error item to estimate the gradient error of the parabolic correction term
    # jax.debug.print('{}',jnp.nansum(e4)))
    e_tot=e1+e2+e3+e4
    return e_tot,jnp.nansum(dAp)#抛物线近似的补偿项
@jax.jit
def error_critial(i,create,Is_create,parity,deXProde2X,z,de_z):
    pos_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==-1*create),size=1)[0]
    neg_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==1*create),size=1)[0]
    z_pos=z[i,pos_idx]
    z_neg=z[i,neg_idx]
    theta_wave=jnp.abs(z_pos-z_neg)/jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx])))
    ce1=1/48*jnp.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3
    ce2=3/2*jnp.abs(dot_product(z_pos-z_neg,de_z[i,pos_idx]-de_z[i,neg_idx])-create*2*jnp.abs(z_pos-z_neg)*jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx]))))*theta_wave
    dAcP=parity[i,pos_idx]*1/24*(deXProde2X[i,pos_idx]-deXProde2X[i,neg_idx])*theta_wave**3
    ce3=1/10*jnp.abs(dAcP)*theta_wave**2
    ce_tot=ce1+ce2+ce3
    return ce_tot,jnp.nansum(dAcP),1/2*(z[i,pos_idx].imag+z[i,neg_idx].imag)*(z[i,pos_idx].real-z[i,neg_idx].real)#critial 附近的抛物线近似'''
@jax.jit
def error_sum(Is_create,z,parity,theta,rho,q,s):
    error_hist=jnp.zeros_like(theta)
    deXProde2X,de_z,delta_theta,de_deXPro_de2X=basic_partial(z,theta,rho,q,s)
    mag=jnp.array([0.])
    e_ord,parab=error_ordinary(deXProde2X,de_z,delta_theta,z,parity,de_deXPro_de2X)
    error_hist=error_hist.at[1:].set(e_ord[:,None])
    carry=jax.lax.cond((Is_create!=0).any(),no_create_true_fun,lambda x:x,(mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z))
    mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z=carry
    #mag=jnp.where(jnp.isnan(mag),0.,mag)
    error_hist=error_hist.at[-2:].set(0.)
    return error_hist/(np.pi*rho**2),mag,parab
@jax.jit
def no_create_true_fun(carry):
    ## if there is image create or destroy, we need to calculate the error
    mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z=carry
    critial_idx_row=jnp.where((Is_create!=0).any(axis=1),size=20,fill_value=-2)[0]
    carry,_=jax.lax.scan(error_scan,(mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z),critial_idx_row)
    return carry
@jax.jit
def error_scan(carry,i):
    mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z=carry
    carry=jax.lax.cond(jnp.abs(Is_create[i].sum())/2==1,create_in_diff_row,create_in_same_row,(mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i))
    return carry[0:-1],i
@jax.jit
def create_in_diff_row(carry):
    # if the creation and destruction of the image are in different rows 3-5-5-3
    mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i=carry
    create=Is_create[i].sum()/2
    e_crit,dacp,magc=error_critial(i,create,Is_create,parity,deXProde2X,z,de_z)
    error_hist=error_hist.at[(i-(create-1)/2).astype(int)].add(e_crit)
    magc=jnp.where(jnp.isnan(magc),0.,magc)
    mag+=magc
    parab+=dacp
    return  (mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i)
@jax.jit
def create_in_same_row(carry):
    # if the creation and destruction of the image are in the same row 3-5-3
    mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i=carry
    create=1
    e_crit,dacp,magc=error_critial(i,create,Is_create,parity,deXProde2X,z,de_z)
    error_hist=error_hist.at[(i-(create-1)/2).astype(int)].add(e_crit)
    parab+=dacp
    magc=jnp.where(jnp.isnan(magc),0.,magc)
    mag+=magc
    create=-1
    e_crit,dacp,magc=error_critial(i,create,Is_create,parity,deXProde2X,z,de_z)
    magc=jnp.where(jnp.isnan(magc),0.,magc)
    error_hist=error_hist.at[(i-(create-1)/2).astype(int)].add(e_crit)
    parab+=dacp
    mag+=magc
    return  (mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i)