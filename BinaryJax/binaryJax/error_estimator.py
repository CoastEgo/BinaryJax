import numpy as np
import jax.numpy as jnp
import jax
from .basic_function_jax import dot_product,basic_partial
from .util import stop_grad_wrapper
@jax.jit
def error_ordinary(deXProde2X,de_z,delta_theta,z,parity,de_deXPro_de2X):
    e1=jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta**3)
    dAp_1=1/24*((deXProde2X[0:-1]+deXProde2X[1:]))*delta_theta
    dAp=dAp_1*delta_theta**2*parity[0:-1]
    delta_theta_wave=jnp.abs(z[0:-1]-z[1:])**2/jnp.abs(dot_product(de_z[0:-1],de_z[1:]))
    e2=3/2*jnp.abs(dAp_1*(delta_theta_wave-delta_theta**2))
    e3=1/10*jnp.abs(dAp)*delta_theta**2
    de_dAp=1/24*(de_deXPro_de2X[0:-1]-de_deXPro_de2X[1:])*delta_theta**3*parity[0:-1]
    e4 = 1/10*jnp.abs(de_dAp) ## e4 is the error item to estimate the gradient error of the parabolic correction term
    # jax.debug.print('{}',jnp.nansum(e4)))
    e_tot=e1+e2+e3+e4
    return e_tot,dAp#抛物线近似的补偿项
@jax.jit
def error_critial(pos_idx,neg_idx,i,create,parity,deXProde2X,z,de_z):
    # pos_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==-1*create),size=1)[0]
    # neg_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==1*create),size=1)[0]
    z_pos=z[i,pos_idx]
    z_neg=z[i,neg_idx]
    theta_wave=jnp.abs(z_pos-z_neg)/jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx])))
    ce1=1/48*jnp.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3
    ce2=3/2*jnp.abs(dot_product(z_pos-z_neg,de_z[i,pos_idx]-de_z[i,neg_idx])-create*2*jnp.abs(z_pos-z_neg)*jnp.sqrt(jnp.abs(dot_product(de_z[i,pos_idx],de_z[i,neg_idx]))))*theta_wave
    dAcP=parity[i,pos_idx]*1/24*(deXProde2X[i,pos_idx]-deXProde2X[i,neg_idx])*theta_wave**3
    ce3=1/10*jnp.abs(dAcP)*theta_wave**2
    ce_tot=ce1+ce2+ce3
    return ce_tot,jnp.sum(dAcP),1/2*(z[i,pos_idx].imag+z[i,neg_idx].imag)*(z[i,pos_idx].real-z[i,neg_idx].real)#critial 附近的抛物线近似'''
@jax.jit
def error_sum(Roots_State,rho,q,s,mask=None):
    Is_create = Roots_State.Is_create
    z = Roots_State.roots
    parity = Roots_State.parity
    theta = Roots_State.theta
    if mask is None:
        mask = ~jnp.isnan(z)
    caustic_crossing = (Is_create!=0).any()
    deXProde2X,de_z,de_deXPro_de2X=basic_partial(z,theta,rho,q,s,caustic_crossing)
    deXProde2X = jnp.where(mask,deXProde2X,0.)
    de_z = jnp.where(mask,de_z,0.)
    de_deXPro_de2X = jnp.where(mask,de_deXPro_de2X,0.)

    error_hist=jnp.zeros_like(theta)
    delta_theta = jnp.diff(theta,axis=0)
    
    mag=jnp.array([0.])
    e_ord,parab=error_ordinary(deXProde2X,de_z,delta_theta,z,parity,de_deXPro_de2X)

    diff_mask = mask[1:] & mask[:-1]
    e_ord = jnp.where(diff_mask,e_ord,0.)
    parab = jnp.where(diff_mask,parab,0.)
    e_ord = jnp.sum(e_ord,axis=1)
    parab = jnp.sum(parab)

    error_hist=error_hist.at[1:].set(e_ord[:,None])

    de_z = jnp.where(mask,de_z,10.) # avoid the nan value in the theta_wave calculation
    @jax.jit
    def no_create_true_fun(carry):
        ## if there is image create or destroy, we need to calculate the error

        mag,parab,error_hist=carry
        critial_idx_row=jnp.where((Is_create!=0).any(axis=1),size=20,fill_value=-2)[0]
        result_row_idx = jnp.zeros_like(critial_idx_row)
        critial_pos_idx=jnp.zeros_like(critial_idx_row)
        critial_neg_idx=jnp.zeros_like(critial_idx_row)
        create_array = jnp.zeros_like(critial_idx_row)
        total_num = 0
        # carry,_=jax.lax.scan(error_scan,(Is_create,parity,
        #                                  result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num),critial_idx_row)
        carry_input = (Is_create,parity,result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num)
        scan_fun = lambda x : jax.lax.scan(error_scan,x,critial_idx_row)[0]
        carry_output = stop_grad_wrapper(scan_fun)(carry_input)
        Is_create_temp,parity_temp,result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num=carry_output
        critical_error,dApc,magc = jax.vmap(error_critial,in_axes=(0,0,0,0,None,None,None,None))(
            critial_pos_idx,critial_neg_idx,result_row_idx,create_array,parity,deXProde2X,z,de_z)
        critical_error = jnp.where(jnp.arange(len(critical_error))<total_num,critical_error,0.)
        magc = jnp.where(jnp.arange(len(magc))<total_num,magc,0.)
        dApc = jnp.where(jnp.arange(len(dApc))<total_num,dApc,0.)

        error_hist=error_hist.at[(result_row_idx-(create_array-1)//2),0].add(critical_error)
        mag+=jnp.sum(magc)
        parab+=jnp.sum(dApc)

        return (mag,parab,error_hist)
    carry=jax.lax.cond(caustic_crossing,no_create_true_fun,lambda x:x,(0.,0.,jnp.zeros_like(theta)))
    mag_c,parab_c,error_hist_c=carry
    mag+=mag_c
    parab+=parab_c
    error_hist+=error_hist_c

    return error_hist/(np.pi*rho**2),mag,parab
@stop_grad_wrapper
@jax.jit
def error_scan(carry,i):
    Is_create,parity,result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num=carry
    @jax.jit
    def create_in_diff_row(carry):
        # if the creation and destruction of the image are in different rows 3-5-5-3
        result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i=carry
        create=(Is_create[i].sum()//2)
        # e_crit,dacp,magc=error_critial(i,create,Is_create,parity,deXProde2X,z,de_z)
        pos_idx_i,neg_idx_i = find_pos_idx(i,create,Is_create,parity)
        result_row_idx= result_row_idx.at[total_num].set(i)
        critial_pos_idx= critial_pos_idx.at[total_num].set(pos_idx_i[0])
        critial_neg_idx= critial_neg_idx.at[total_num].set(neg_idx_i[0])
        create_array= create_array.at[total_num].set(create)

        total_num+=1
        # error_hist=error_hist.at[(i-(create-1)/2).astype(int)].add(e_crit)
        # magc=jnp.where(jnp.isnan(magc),0.,magc)
        # mag+=magc
        # parab+=dacp
        return (result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i)
    @jax.jit
    def create_in_same_row(carry):
        # if the creation and destruction of the image are in the same row 3-5-3
        result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i=carry
        # e_crit,dacp,magc=error_critial(i,create,Is_create,parity,deXProde2X,z,de_z)
        # error_hist=error_hist.at[(i-(create-1)/2).astype(int)].add(e_crit)
        create = 1
        pos_idx_i,neg_idx_i = find_pos_idx(i,create,Is_create,parity)
        result_row_idx= result_row_idx.at[total_num].set(i)
        critial_pos_idx= critial_pos_idx.at[total_num].set(pos_idx_i[0])
        critial_neg_idx= critial_neg_idx.at[total_num].set(neg_idx_i[0])
        create_array= create_array.at[total_num].set(create)
        total_num+=1
        
        create = -1
        pos_idx_i,neg_idx_i = find_pos_idx(i,create,Is_create,parity)
        result_row_idx= result_row_idx.at[total_num].set(i)
        critial_pos_idx= critial_pos_idx.at[total_num].set(pos_idx_i[0])
        critial_neg_idx= critial_neg_idx.at[total_num].set(neg_idx_i[0])
        create_array= create_array.at[total_num].set(create)
        total_num+=1
        return  (result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i)
    not_nan_function = lambda x: jax.lax.cond(jnp.abs(Is_create[i].sum())/2==1,create_in_diff_row,create_in_same_row,x)
    carry = jax.lax.cond(Is_create[i].sum()!=0,not_nan_function,lambda x:x,(result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i))
    (result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num,i)=carry

    carry = (Is_create,parity,result_row_idx,critial_pos_idx,critial_neg_idx,create_array,total_num)
    # carry=jax.lax.cond(jnp.abs(Is_create[i].sum())/2==1,create_in_diff_row,create_in_same_row,(mag,parab,Is_create,error_hist,parity,deXProde2X,z,de_z,i))
    return carry,i
@jax.jit
def find_pos_idx(i,create,Is_create,parity):
    """
    find the positive and negative index for image creation and destruction
    this index is used to determine the order for trapzoidal integration: 
        for image creation, the integration should be  (z.imag_- + z.imag_+)*(z.real_-  - z.real_+)
        for image destruction, the integration should be (z.imag_- + z.imag_+)*(z.real_+  - z.real_-)
    the create represents the flag for image creation or destruction : 
        if create=1, it means image creation, if create=-1, it means image destruction
        and for image creation, the error should be added to the previous row, for image destruction, the error should be added to the next row
    
    """
    pos_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==-1*create),size=1)[0]
    neg_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==1*create),size=1)[0]
    return pos_idx,neg_idx