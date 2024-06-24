import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from .error_estimator import *
from .solution import *
from .basic_function_jax import Quadrupole_test
from functools import partial
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@jax.jit
def point_light_curve(trajectory_l,s,q,m1,m2,rho):
    zeta_l = trajectory_l[:,None]
    coff=get_poly_coff(zeta_l,s,m2)
    z_l=get_roots_vmap(trajectory_l.shape[0],coff)
    error=verify(zeta_l,z_l,s,m1,m2)
    cond=error<1e-6
    index=jnp.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5),size=5,fill_value=-1)[0]
    def ambigious_deal(carry):
        error,index,cond=carry
        sortidx=jnp.argsort(error[index],axis=1)
        cond=cond.at[index].set(False)
        cond=cond.at[index[:,None],sortidx[:,0:3]].set(True)
        return cond
    cond=lax.cond((index!=-1).any(),ambigious_deal,lambda x : x[-1],(error,index,cond)) #某些情况下出现的根不是5或者3个#'''

    z=jnp.where(cond,z_l,jnp.nan)
    zG=jnp.where(cond,jnp.nan,z_l)
    cond,mag=Quadrupole_test(rho,s,q,zeta_l,z,zG)
    return mag,cond
    
@partial(jax.jit,static_argnames=['return_info','default_strategy'])
def model(t_0,u_0,t_E,rho,q,s,alpha_deg,times,tol=1e-2,retol=0.001,return_info=False,default_strategy=(60,80,150)):
    # Here the parameterization is consistent with Mulensmodel and VBBinaryLensing
    # But the alpha is 180 degree different from VBBinaryLensing
    ### initialize parameters
    alpha_rad=alpha_deg*2*jnp.pi/360
    tau=(times-t_0)/t_E
    trajectory_n=tau.shape[0]
    m1=1/(1+q);m2=q/(1+q)
    ## switch the coordinate system to the lowmass
    trajectory = tau*jnp.exp(1j*alpha_rad)+1j*u_0*jnp.exp(1j*alpha_rad)
    trajectory_l = to_lowmass(s,q,trajectory)

    mag,cond=point_light_curve(trajectory_l,s,q,m1,m2,rho)

    if return_info:
        pad_value = [jnp.nan,0.,jnp.nan+1j*jnp.nan,jnp.nan,jnp.nan,0.,True,0]
        shape = [1,1,5,5,1,1,1,5]
        init_fun = lambda x,y : jnp.full((sum(default_strategy),y),x)
        theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,Is_create = jax.tree_map(init_fun,pad_value,shape)
        sample_n=jnp.array([0]);epsilon=tol;epsilon_rel=retol;mag_no_diff_num=0;outloop=0
        carry_init = (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l[0],rho,s,q,epsilon,epsilon_rel,jnp.array([0.]),mag_no_diff_num,outloop)
        
        mag_contour = lambda trajectory_l: heriachical_contour(trajectory_l,tol,retol,rho,s,q,m1,m2,trajectory_n,default_strategy)
        result = lax.map(lambda x: lax.cond(x[0],lambda _: (x[1],carry_init), jax.jit(mag_contour),x[2]), [cond,mag,trajectory_l])
        mag_final,carry_list = result
        return mag_final,carry_list
    else:
        mag_contour = lambda trajectory_l: heriachical_contour(trajectory_l,tol,retol,rho,s,q,m1,m2,trajectory_n,default_strategy)[0]
        mag_final = lax.map(lambda x: lax.cond(x[0],lambda _: x[1], 
                                               jax.jit(mag_contour),x[2]), 
                                               [cond,mag,trajectory_l])

        return mag_final

def to_centroid(s,q,x):#change coordinate system to cetorid
    delta_x=s/(1+q)
    return -(jnp.conj(x)-delta_x)
def to_lowmass(s,q,x):#change coordinaate system to lowmass
    delta_x=s/(1+q)
    return -jnp.conj(x)+delta_x
@partial(jax.jit,static_argnames=['default_strategy'])
def heriachical_contour(trajectory_l,tol,retol,rho,s,q,m1,m2,sample_n,default_strategy=(60,80,150)):
    # JIT compile operation needs shape of the array to be determined.
    # But for optimial sampling, It is hard to know the array length before the code runs so we need to assign large enough array length
    # which will cause the waste of memory and time. 
    # To solve this problem, here we use heriachical array length adding method to add array length gradually,
    # the problem is we should fine tuen the array length added in different layers to get the optimal performance which depends on the tolerance and parameter.
    # current is 60 + 80 + 150 = 290
    @partial(jax.jit,static_argnums=(-1,))
    def reshape_fun(carry,arraylength):
        (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)=carry
        ## reshape the array and fill the new array with nan
        pad_list = [theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,Is_create]
        pad_value = [jnp.nan,0.,jnp.nan,jnp.nan,jnp.nan,0.,True,0]
        padded_list =jax.tree_map(lambda x,y: jnp.pad(x,((0,arraylength),(0,0)),'constant',constant_values=y),pad_list,pad_value)

        theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,Is_create=padded_list
        carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)
        return carry
    @jax.jit
    def secondary_contour(carry):
        result,resultlast,add_length,Max_array_length=carry

        ## switch the different method to add points while loop or scan
        ## while loop don't support the reverse mode differentiation and shard_map in current jax version

        ## while loop 
        resultnew,resultlast=lax.while_loop(cond_fun,while_body_fun,(resultlast,resultlast))
        ## scan
        '''resultnew,_=lax.scan(scan_body,(resultlast,resultlast),jnp.arange(5))
        resultnew=resultnew[0]'''
        
        Max_array_length+=add_length
        return resultnew,resultlast,Max_array_length
    # first add
    result=contour_integrate(rho,s,q,m1,m2,trajectory_l,tol,epsilon_rel=retol,inite=30,n_ite=default_strategy[0])
    result,resultlast=result
    Max_array_length=default_strategy[0]
    for i in range(len(default_strategy)-1):
        sample_n = result[0]
        add_length = default_strategy[i+1]

        resultlast = reshape_fun(resultlast,add_length)
        result = reshape_fun(result,add_length)

        result,resultlast,Max_array_length=lax.cond((result[0]<Max_array_length-5)[0],lambda x:(x[0],x[1],x[-1]),secondary_contour,(result,resultlast,add_length,Max_array_length))

    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
    Is_create,_,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)=result
    maglast = resultlast[-3]
    mag=lax.cond((sample_n<Max_array_length-5)[0],lambda x:x[0],lambda x:x[1],(mag,maglast))

    return (mag[0],result)

@partial(jax.jit,static_argnames=('inite','n_ite'))
def contour_integrate(rho,s,q,m1,m2,trajectory_l,epsilon,epsilon_rel=0,inite=30,n_ite=60):
    ###初始化
    sample_n=jnp.array([inite])
    theta=jnp.where(jnp.arange(n_ite)<inite,jnp.resize(jnp.linspace(0,2*jnp.pi,inite),n_ite),jnp.nan)[:,None]#shape(500,1)
    error_hist=jnp.ones(n_ite)
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    coff=get_poly_coff(zeta_l,s,q/(1+q))
    roots,parity,ghost_roots_dis,outloop,coff,zeta_l,theta=get_real_roots(coff,zeta_l,theta,s,m1,m2)
    buried_error=get_buried_error(ghost_roots_dis,sample_n)
    sort_flag=jnp.where(jnp.arange(n_ite)<inite,False,True)[:,None]#是否需要排序
    ### no need to sort first idx
    sort_flag=sort_flag.at[0].set(True)
    roots,parity,sort_flag=get_sorted_roots(roots,parity,sort_flag)
    Is_create=find_create_points(roots,sample_n)
    #####计算第一次的误差，放大率
    mag_no_diff_num = 0
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
            Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)
    carrylast=carry

    ## switch the different method to add points while loop or scan

    result=lax.while_loop(cond_fun,while_body_fun,(carry,carrylast))

    #result,_=lax.scan(scan_body,(carry,carrylast),jnp.arange(10))

    return result

def scan_body(carry,i):
    # function to adaptively add points using scan with a fixed number of loops
    # prepare for the reverse mode differentiation or shard_map(it is not compatible with while_loop)
    carry=jax.lax.cond(cond_fun(carry),while_body_fun,lambda x:x,carry)
    return carry,i
@jax.jit
def cond_fun(carry):
    carry,carrylast=carry
    ## function to judge whether to continue the loop use relative error
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)=carry
    Max_array_length=jnp.shape(theta)[0]
    mini_interval=jnp.nanmin(jnp.abs(jnp.diff(theta,axis=0)))
    abs_mag_cond=(jnp.nansum(error_hist)>epsilon)

    abs_mag_cond2=(error_hist>epsilon/jnp.sqrt(sample_n)).any()
    rel_mag_cond=(error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n)).any()

    # rel_mag_cond=(jnp.nansum(error_hist)>epsilon_rel*mag)[0]
    # relmag_diff_cond=(jnp.abs((mag-maglast)/maglast)>1/2*epsilon_rel)[0]
    # mag_diff_cond=(jnp.abs(mag-maglast)>1/2*epsilon)[0]

    ## switch the different stopping condition: absolute error or relative error
    ## to modify the stopping condition, you will also need to modify the add points method in the while_body_fun
    # outloop is the number of loop whose add points have ambiguous parity or roots, in this situation we will delete this points and add outloop by 1,
    # if outloop is larger than the threshold we stop the while loop

    loop= (rel_mag_cond& (mini_interval>1e-14)& (outloop<=2)& abs_mag_cond & (mag_no_diff_num<3) & (sample_n<Max_array_length-5)[0])
    # jax.debug.print('{}',mag)
    # jax.debug.breakpoint()
    #loop= ((rel_mag_cond ) & (mini_interval>1e-14)& (~outloop)& abs_mag_cond  & (sample_n<Max_array_length-5)[0])
    #loop= (abs_mag_cond2&(mini_interval>1e-14)& (~outloop)& abs_mag_cond & (mag_diff_cond|(sample_n<Max_array_length/2)[0]) & (sample_n<Max_array_length-5)[0])
    return loop
@jax.jit
def while_body_fun(carry):
    carry,carrylast=carry
    carrylast=carry
    ## function to add points, calculate the error and mag
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)=carry
    add_max=4
    Max_array_length=jnp.shape(theta)[0]

    #一次多个区间加点:
    
    ### absolute error adding mode

    #idx=jnp.where(error_hist>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length/5),fill_value=0)[0]
    #add_number=jnp.ceil((error_hist[idx]/epsilon_rel*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    
    # relative error adding mode
    # error_hist_sorted = jnp.sort(error_hist,axis=0)[::-1]
    # sort_idx = jnp.argsort(error_hist,axis=0)[::-1]

    # idx=jnp.where(error_hist_sorted/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length),fill_value=-1)[0]
    # #print('idx', idx_2)

    # idx = sort_idx[idx].reshape(-1)
    # idx = jnp.sort(idx)
    # zerot_counts = jnp.sum(idx==0)
    # idx = jnp.roll(idx,-zerot_counts)

    idx = jnp.where(error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length/5),fill_value=0)[0]
    
    add_number=jnp.ceil((error_hist[idx]/jnp.abs(mag)/epsilon_rel*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    
    add_number=jnp.where((idx==0)[:,None],0,add_number)
    add_number=jnp.where(add_number>add_max,add_max,add_number)
    
    carry,_=lax.scan(theta_encode,(theta,idx,add_number,
                               jnp.full(theta.shape,jnp.nan)),jnp.arange(idx.shape[0]))
    add_theta=carry[-1]
    ####
    add_zeta_l=get_zeta_l(rho,trajectory_l,add_theta)
    add_coff=get_poly_coff(add_zeta_l,s,q/(1+q))
    sample_n+=jnp.sum(add_number)
    theta,ghost_roots_dis,buried_error,sort_flag,roots,parity,Is_create,add_outloop=add_points(
        idx,add_zeta_l,add_coff,add_theta,roots,parity,theta,ghost_roots_dis,sort_flag,s,1/(1+q),q/(1+q),sample_n,add_number)
    outloop += add_outloop
    ## refine gradient of roots respect to zeta_l
    # zeta_l=get_zeta_l(rho,trajectory_l,theta)
    # roots = refine_gradient(zeta_l,q,s,roots)
    maglast=mag
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    mag_no_diff_num += (jnp.abs(mag-maglast)<1/2*epsilon).sum()
    # check the change of the mag, if the mag is not changed at least 2 iteration, we stop the loop
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,mag_no_diff_num,outloop)
    return (carry,carrylast)
@jax.custom_jvp
def refine_gradient(zeta_l,q,s,z):
    return z
@refine_gradient.defjvp
def refine_gradient_jvp(primals,tangents):
    '''
    use the custom jvp to refine the gradient of roots respect to zeta_l, based on the equation on V.Bozza 2010 eq 20. The necessity of this function is still under investigation.
    '''
    zeta,q,s,z=primals
    tangent_zeta,tangent_q,tangent_s,tangent_z=tangents

    z_c=jnp.conj(z)
    parZetaConZ=1/(1+q)*(1/(z_c-s)**2+q/z_c**2)
    detJ = 1-jnp.abs(parZetaConZ)**2

    add_item_q =  1/(1+q)**2*(1/(z_c-s)-1/z_c)
    add_item_q = tangent_q*(add_item_q-jnp.conj(add_item_q)*parZetaConZ)

    add_item_s = -1/(1+q)/(z_c-s)**2
    add_item_s = tangent_s*(add_item_s-jnp.conj(add_item_s)*parZetaConZ)

    tangent_z2 =  (tangent_zeta-parZetaConZ * jnp.conj(tangent_zeta)-add_item_q-add_item_s)/detJ
    tangent_z2 = jnp.where(jnp.isnan(tangent_z2),0.,tangent_z2)
    # jax.debug.print('{}',(tangent_z2-tangent_z).sum())
    return z,tangent_z2
