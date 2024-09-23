import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from .error_estimator import *
from .solution import *
from .basic_function_jax import Quadrupole_test,to_centroid,to_lowmass,refine_gradient
from functools import partial
from .util import Iterative_State,Error_State,Model_Param,insert_body,stop_grad_wrapper,warn_length_not_enough
from .polynomial_solver import get_roots_vmap
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@partial(jax.jit,static_argnames=['return_num'])
def point_light_curve(trajectory_l,s,q,rho,tol,return_num=False):
    """
    Calculate the point source light curve.

    Args:
        trajectory_l (ndarray): The trajectory of the lensing event.
        s (float): The projected separation between the lens and the source.
        q (float): The mass ratio between the lens and the source.
        rho (float): The source radius in units of the Einstein radius.
        tol (float): The absolute tolerance for the quadrupole test.
        return_num (bool, optional): Whether to return the number of real roots. Defaults to False.

    Returns:
        tuple: A tuple containing the magnitude of the light curve and a boolean array indicating the validity of the calculation.
        if the quadrupole test is passed, the corresponding element in the boolean array is True. If the return_num is True, the tuple will also contain the number of real roots.
    """
    m1=1/(1+q);m2=q/(1+q)
    zeta_l = trajectory_l[:,None]
    coff=get_poly_coff(zeta_l,s,m2)
    z_l=get_roots_vmap(trajectory_l.shape[0],coff)
    error=verify(zeta_l,z_l,s,m1,m2)

    iterator = jnp.arange(zeta_l.shape[0]) # criterion to select the roots, same as the VBBL
    dlmin = 1.0e-4; dlmax = 1.0e-3
    sort_idx=jnp.argsort(error,axis=1)
    third_error = error[iterator,sort_idx[:,2]]
    forth_error = error[iterator,sort_idx[:,3]]
    three_roots_cond = (forth_error*dlmin) > (third_error+1e-12) # three roots criterion
    # bad_roots_cond = (~three_roots_cond) & ((forth_error*dlmax) > (third_error+1e-12)) # bad roots criterion
    mask = jnp.ones_like(z_l,dtype=bool)
    full_value = jnp.where(three_roots_cond,False,True)
    mask = mask.at[iterator[:,None],sort_idx[:,3:]].set(full_value[:,None])

    cond,mag=Quadrupole_test(rho,s,q,zeta_l,z_l,mask,tol)
    if return_num:
        return mag,cond,mask.sum(axis=1)
    else:
        return mag,cond
    
@partial(jax.jit,static_argnames=['return_info','default_strategy','analytic'])
def model(t_0,u_0,t_E,rho,q,s,alpha_deg,times,tol=1e-2,retol=0.001,return_info=False,default_strategy=(35,30,60,120,240,480),analytic=True):
    """
    Compute the microlensing model for a binary lens system using JAX.

    Args:
        t_0 (float): The time of the peak of the microlensing event.
        u_0 (float): The impact parameter of the source trajectory.
        t_E (float): The Einstein crossing time.
        rho (float): The source radius normalized to the Einstein radius.
        q (float): The planet to host mass ratio of the binary lens system.
        s (float): The projected separation of the binary lens system normalized to the Einstein radius.
        alpha_deg (float): The angle between the source trajectory and the binary axis in degrees.
        times (array-like): The times at which to compute the model.
        tol (float, optional): The tolerance for the adaptive contour integration. Defaults to 1e-2.
        retol (float, optional): The relative tolerance for the adaptive contour integration. Defaults to 0.001.
        return_info (bool, optional): Whether to return additional information about the computation. Defaults to False.
        default_strategy (tuple, optional): The default strategy for the hierarchical contour integration. Defaults to (60,80,150).
        analytic (bool, optional): Whether to use the analytic chain rule to simplify the computation graph. Set this to True will accelerate 
        the computation of the gradient and will support the reverse mode differentiation containing the while loop. But set this to True will slow down if only
        calculate the model without differentiation. Defaults to True.
    Returns:
        array-like: The magnification of the source at the given times.
        tuple: Additional information about the computation if return_info is True.
    """
    # Here the parameterization is consistent with Mulensmodel and VBBinaryLensing
    ### initialize parameters
    alpha_rad=alpha_deg*2*jnp.pi/360
    tau=(times-t_0)/t_E
    ## switch the coordinate system to the lowmass
    trajectory = tau*jnp.exp(1j*alpha_rad)+1j*u_0*jnp.exp(1j*alpha_rad)
    trajectory_l = to_lowmass(s,q,trajectory)

    mag,cond=point_light_curve(trajectory_l,s,q,rho,tol)

    if return_info:
        pad_value = [jnp.nan,0.,jnp.nan+1j*jnp.nan,jnp.nan,jnp.nan,0.,True]
        shape = [1,1,5,5,1,1,1]
        init_fun = lambda x,y : jnp.full((sum(default_strategy),y),x)
        theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag = jax.tree_map(init_fun,pad_value,shape)
        sample_n=jnp.array([0])
        roots_state = Iterative_State(sample_n,theta,roots,parity,ghost_roots_dis,sort_flag)
        mag_contour = lambda trajectory_l: contour_integral(trajectory_l,tol,retol,rho,s,q,default_strategy,analytic)
        result = lax.map(lambda x: lax.cond(x[0],lambda _: (x[1],(x[2],rho,s,q,
                                                            roots_state,Error_State(jnp.array([x[1]]),0,0,error_hist,tol,retol))
                                                            ), jax.jit(mag_contour),x[2]), [cond,mag,trajectory_l])
        mag_final,carry_list = result
        return mag_final,carry_list
    else:
        mag_contour = lambda trajectory_l: contour_integral(trajectory_l,tol,retol,rho,s,q,default_strategy,analytic)[0]
        mag_final = lax.map(lambda x: lax.cond(x[0],lambda _: x[1], 
                                               jax.jit(mag_contour),x[2]), 
                                               [cond,mag,trajectory_l])

        return mag_final

def anayltic_warpper(trajectory_l,rho,s,q,roots_State,mag_State):
    """
    Wrapper function for the hierarchical contour integration.
    This function is used for better support of automatic differentiation, which can reduce the computational graph size to accelerate the computation and
    support the reverse mode differentiation containing the while loop.
    Args:
        trajectory_l (complex): The trajectory of the lensing event at the low mass coordinate system.
        tol (float): The tolerance value.
        retol (float): The relative tolerance value.
        rho (float): The density value.
        s (float): The separation value.
        q (float): The mass ratio value
    """

    sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create=roots_State
    mask = ~jnp.isnan(roots)
    roots_100fill = jnp.where(mask,roots,100.)
    parity = jnp.where(mask,parity,0.)
    theta = jnp.where(mask,theta,0.)

    # stop gradient to avoid nan in reverse mode
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    roots_100fill = refine_gradient(zeta_l,q,s,roots_100fill)

    mask_diff = mask[1:] & mask[:-1]
    roots_State_refine_grad = Iterative_State(sample_n,theta,roots_100fill,parity,ghost_roots_dis,sort_flag,Is_create)
    mag_ndarray= (roots_100fill.imag[0:-1]+roots_100fill.imag[1:])*(roots_100fill.real[0:-1]-roots_100fill.real[1:])*parity[0:-1]
    mag = 1/2*jnp.sum(jnp.where(mask_diff,mag_ndarray,0.).sum(axis=1))
    
    _,magc,parab=error_sum(roots_State_refine_grad,rho,q,s,mask)
    # parab = jax.lax.stop_gradient(parab)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)

    mag_State = mag_State._replace(mag=mag)
    return (trajectory_l,rho,s,q,roots_State,mag_State)
@partial(jax.jit,static_argnames=['default_strategy','analytic'])
def contour_integral(trajectory_l,tol,retol,rho,s,q,default_strategy=(60,80,150),analytic=True):
    """
    Perform hierarchical sampling for adaptive contour integration. 
    This function is used to reduce the memory usage and improve the performance of the contour integration. The reason is that the optimal fixed array length 
    is hard to determine before the code runs which the basic requirement for JIT compilation.

    Args:
        trajectory_l (complex): The trajectory of the lensing event at the low mass coordinate system.
        tol (float): The tolerance value.
        retol (float): The relative tolerance value.
        rho (float): The density value.
        s (float): The separation value.
        q (float): The mass ratio value.
        default_strategy (tuple, optional): The default strategy for array length. Defaults to (60,80,150).

    Returns:
        tuple: A tuple containing the magnitude and the result of the contour integration.
    """
    # JIT compile operation needs shape of the array to be determined.
    # But for optimial sampling, It is hard to know the array length before the code runs so we need to assign large enough array length
    # which will cause the waste of memory and time. 
    # To solve this problem, here we use heriachical array length adding method to add array length gradually,
    # the problem is we should fine tuen the array length added in different layers to get the optimal performance which depends on the tolerance and parameter.
    # current is 60 + 80 + 150 = 290
    @partial(jax.jit,static_argnums=(-1,))
    def reshape_fun(carry,arraylength):
        """
        Reshape the arrays and fill the new array with NaN values.

        Args:
            carry: The carry variable.
            arraylength (int): The length of the array to be added.

        Returns:
            The reshaped carry variable.
        """
        (trajectory_l,rho,s,q,roots_State,mag_State)=carry

        sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create=roots_State

        error_hist = mag_State.error_hist
        ## reshape the array and fill the new array with nan
        pad_list = [theta,error_hist,roots,parity,ghost_roots_dis,sort_flag]
        pad_value = [jnp.nan,0.,jnp.nan,jnp.nan,jnp.nan,True]
        pad_fun = lambda x,y: jnp.pad(x,((0,arraylength),(0,0)),'constant',constant_values=y)
        if analytic:
            pad_fun = stop_grad_wrapper(pad_fun)
            padded_list = jax.tree_map(pad_fun,pad_list,pad_value)
            padded_list = jax.lax.stop_gradient(padded_list)
        else:
            padded_list =jax.tree_map(pad_fun,pad_list,pad_value)
        
        theta,error_hist,roots,parity,ghost_roots_dis,sort_flag=padded_list
        carry=(trajectory_l,rho,s,q,
               Iterative_State(sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create),
                Error_State(mag_State.mag,mag_State.mag_no_diff,mag_State.outloop,error_hist,mag_State.epsilon,mag_State.epsilon_rel))
        return carry
    
    @jax.jit
    def secondary_contour(carry):
        """
        Perform secondary contour integration with a longer array length.

        Args:
            carry: The carry variable.

        Returns:
            The result of the secondary contour integration.
        """
        result,resultlast,add_length,Max_array_length=carry

        ## switch the different method to add points while loop or scan
        ## while loop don't support the reverse mode differentiation and shard_map in current jax version

        ## while loop 

        # resultnew,resultlast=lax.while_loop(cond_fun,while_body_fun,(resultlast,resultlast))
        if analytic:
            stop_grad_loop = stop_grad_wrapper(lambda x:lax.while_loop(cond_fun,while_body_fun,x))
            resultnew,resultlast = stop_grad_loop((resultlast,resultlast))
            resultnew = anayltic_warpper(trajectory_l,rho,s,q,resultnew[-2],resultnew[-1])
        else:
            resultnew,resultlast=lax.while_loop(cond_fun,while_body_fun,(resultlast,resultlast))

        Max_array_length+=add_length
        return resultnew,resultlast,Max_array_length
    
    # first add
    
    if analytic:
        carry=stop_grad_wrapper(contour_init)(rho,s,q,trajectory_l,tol,epsilon_rel=retol,inite=30,n_ite=default_strategy[0])
        stop_grad_loop = lambda x:lax.while_loop(cond_fun,while_body_fun,x)
        result_no_grad,resultlast = stop_grad_wrapper(stop_grad_loop)((carry,carry))
        result = anayltic_warpper(trajectory_l,rho,s,q,result_no_grad[-2],result_no_grad[-1])

    else:
        carry=contour_init(rho,s,q,trajectory_l,tol,epsilon_rel=retol,inite=30,n_ite=default_strategy[0])
        result,resultlast=lax.while_loop(cond_fun,while_body_fun,(carry,carry))

    Max_array_length=default_strategy[0]
    for i in range(len(default_strategy)-1):

        add_length = default_strategy[i+1]

        resultlast = reshape_fun(resultlast,add_length)
        result = reshape_fun(result,add_length)

        result,resultlast,Max_array_length=lax.cond((result[-2].sample_num<Max_array_length-2)[0],lambda x:(x[0],x[1],x[-1]),secondary_contour,(result,resultlast,add_length,Max_array_length))

    (trajectory_l,rho,s,q,roots_State,mag_State)=result

    condition = (roots_State.sample_num<Max_array_length-2)[0]
    @jax.jit
    def update_result_fun(carry):
        # update the exceed flag to True in the mag_State
        result_last=carry[1]
        mag_State=result_last[-1]
        jax.debug.callback(warn_length_not_enough,carry[0][-2].sample_num,Max_array_length)
        mag_State_new = mag_State._replace(exceed_flag=True)
        if analytic:
            result_last,mag_State_new = jax.lax.stop_gradient(result_last),jax.lax.stop_gradient(mag_State_new)
            result_last_update = anayltic_warpper(trajectory_l,rho,s,q,result_last[-2],mag_State_new)
        else:
            result_last_update = (trajectory_l,rho,s,q,result_last[-2],mag_State_new)
        return result_last_update
    result = lax.cond(condition,lambda x:x[0],update_result_fun,(result,resultlast))
    return (result[-1].mag[0],result)

@partial(jax.jit,static_argnames=('inite','n_ite'))
def contour_init(rho,s,q,trajectory_l,epsilon,epsilon_rel=0,inite=30,n_ite=60):
    """
    Perform contour integration to calculate the result of the binary lens model.

    Args:
        rho (float): The radius of the lens.
        s (float): The separation between the two lens components.
        q (float): The mass ratio of the two lens components.
        trajectory_l (array): The trajectory of the lens in the low mass coordinate system.
        epsilon (float): The integration precision.
        epsilon_rel (float, optional): The relative integration precision. Defaults to 0.
        inite (int, optional): The number of initial integration points. Defaults to 30.
        n_ite (int, optional): The total number of integration points. Defaults to 60.

    Returns:
        tuple: A tuple containing the integration result and other intermediate variables.
    """
    m1=1/(1+q);m2=q/(1+q)
    sample_n=jnp.array([inite])
    theta=jnp.where(jnp.arange(n_ite)<inite,jnp.resize(jnp.linspace(0,2*jnp.pi,inite),n_ite),jnp.nan)[:,None]#shape(500,1)
    error_hist=jnp.ones(n_ite)
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    coff=get_poly_coff(zeta_l,s,q/(1+q))
    roots,parity,ghost_roots_dis,outloop,coff,zeta_l,theta,_=get_real_roots(coff,zeta_l,theta,s,m1,m2,jnp.arange(n_ite))
    
    buried_error=get_buried_error(ghost_roots_dis,sample_n)/jnp.pi/rho**2
    
    sort_flag=jnp.where(jnp.arange(n_ite)<inite,False,True)[:,None]#是否需要排序
    sort_flag=sort_flag.at[0].set(True) ### no need to sort first idx

    indices_update,sort_flag=get_sorted_roots(roots,parity,sort_flag,n_ite)
    roots = roots[jnp.arange(n_ite)[:,None],indices_update]
    parity = parity[jnp.arange(n_ite)[:,None],indices_update]

    Is_create=find_create_points(roots,parity,sample_n)
    roots_State = Iterative_State(sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create)

    #####计算第一次的误差，放大率
    mag_no_diff_num = 0
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(roots_State,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    mag_State = Error_State(mag,mag_no_diff_num,outloop,error_hist,epsilon,epsilon_rel)

    carry=(trajectory_l,rho,s,q,roots_State,mag_State)

    return carry

@jax.jit
def cond_fun(carry):
    carry,carrylast=carry
    ## function to judge whether to continue the loop use relative error
    (trajectory_l,rho,s,q,roots_State,mag_State)=carry
    theta = roots_State.theta
    sample_n = roots_State.sample_num
    error_hist = mag_State.error_hist
    epsilon = mag_State.epsilon
    epsilon_rel = mag_State.epsilon_rel
    mag = mag_State.mag
    mag_no_diff_num = mag_State.mag_no_diff
    outloop = mag_State.outloop
    K = 2
    Max_array_length=jnp.shape(theta)[0]
    mini_interval=jnp.nanmin(jnp.abs(jnp.diff(theta,axis=0)))
    abs_mag_cond=(jnp.nansum(error_hist)>epsilon)

    # abs_mag_cond2=(error_hist>epsilon/jnp.sqrt(sample_n/2)).any()
    rel_mag_cond=(
        (error_hist/jnp.abs(mag))> 
                  (epsilon_rel/jnp.sqrt(sample_n/K))
                  ).any() # this factor 1/2 is a tunable parameter to adjust the stopping condition, the larger the value, the more strict the stopping condition

    # rel_mag_cond=(jnp.nansum(error_hist)>epsilon_rel*mag)[0]
    # relmag_diff_cond=(jnp.abs((mag-maglast)/maglast)>1/2*epsilon_rel)[0]
    # mag_diff_cond=(jnp.abs(mag-maglast)>1/2*epsilon)[0]

    ## switch the different stopping condition: absolute error or relative error
    ## to modify the stopping condition, you will also need to modify the add points method in the while_body_fun
    # outloop is the number of loop whose add points have ambiguous parity or roots, in this situation we will delete this points and add outloop by 1,
    # if outloop is larger than the threshold we stop the while loop

    loop= (rel_mag_cond& (mini_interval>1e-14)& (outloop<=2)& abs_mag_cond & (mag_no_diff_num<3) & (sample_n<Max_array_length-2)[0])
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
    (trajectory_l,rho,s,q,roots_State,mag_State)=carry
    theta = roots_State.theta
    epsilon_rel = mag_State.epsilon_rel
    error_hist = mag_State.error_hist
    mag = mag_State.mag
    sample_n = roots_State.sample_num

    Max_add=4
    Max_array_length=jnp.shape(theta)[0]
    Max_index_length = Max_array_length//5
    Max_total_num = Max_array_length//2
    K = 2
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

    idx = jnp.where((error_hist/jnp.abs(mag))
                    > (epsilon_rel/jnp.sqrt(sample_n/K)),
                    size=Max_index_length,fill_value=-1)[0]

    add_number=jnp.ceil(
        (error_hist[idx]/jnp.abs(mag)/epsilon_rel*jnp.sqrt(sample_n/K))**0.2
        ).astype(int)#至少要插入一个点，不包括相同的第一个
    
    add_number=jnp.where((idx==-1)[:,None],0,add_number)
    add_number=jnp.where(add_number>Max_add,Max_add,add_number)
    add_number = jax.lax.cond(add_number.sum()>Max_total_num,lambda x : (x*(Max_total_num/x.sum())).astype(int),lambda x:x,add_number)
    # jax.debug.print('add_number {}/{}  idx length {}/{}  sample_n: {}',add_number.sum(),Max_total_num,(idx!=0).sum(),Max_index_length,sample_n[0])
    @jax.jit
    def theta_encode(carry,k):
        (theta,idx,add_number,add_theta_encode)=carry

        theta_diff = (theta[idx[k]] - theta[idx[k]-1]) / (add_number[k]+1)
        add_theta=jnp.arange(1,Max_total_num+1)[:,None]*theta_diff+theta[idx[k]-1]
        add_theta=jnp.where((jnp.arange(Max_total_num)<add_number[k])[:,None],add_theta,jnp.nan)
        carry2,_=insert_body((add_theta_encode,add_theta,jnp.where(jnp.isnan(add_theta_encode),size=1)[0],add_number[k][None]),0)
        add_theta_encode=carry2[0]
        return (theta,idx,add_number,add_theta_encode),k
    @jax.jit
    def update_carry(carrylast):
        carry,_=lax.scan(theta_encode,(theta,idx,add_number,
                                jnp.full((Max_total_num,1),jnp.nan)),jnp.arange(idx.shape[0]))
        add_theta=carry[-1] 
        jax_repeat = jax.jit(jnp.repeat,static_argnames=['axis', 'total_repeat_length'])
        idx_all = jax_repeat(idx, add_number,total_repeat_length=Max_total_num)
        idx_all = jnp.where(jnp.arange(idx_all.shape[0])<add_number.sum(),idx_all,-1)
        ####
        add_zeta_l=get_zeta_l(rho,trajectory_l,add_theta)
        roots_State_new,buried_error,add_outloop=add_points(idx_all,add_zeta_l,add_theta,roots_State,s,1/(1+q),q/(1+q))
        buried_error = buried_error/jnp.pi/rho**2
        mag_State_new = update_mag(roots_State_new,mag_State,rho,q,s,buried_error,add_outloop)
        carry=(trajectory_l,rho,s,q,roots_State_new,mag_State_new)
        return carry
    def no_update_carry(carrylast):
        trajectory_l,rho,s,q,roots_State,mag_State = carrylast
        roots_State_new = roots_State._replace(sample_num=(sample_n+add_number.sum()))
        return (trajectory_l,rho,s,q,roots_State_new,mag_State)
    carry = lax.cond((sample_n[0]+add_number.sum())
                     <(Max_array_length-2),update_carry,no_update_carry,carrylast)
    return (carry,carrylast)

@jax.jit
def update_mag(roots_State,mag_State_last,rho,q,s,buried_error,add_outloop):
    maglast=mag_State_last.mag
    epsilon = mag_State_last.epsilon
    epsilon_rel = mag_State_last.epsilon_rel

    mag=1/2*jnp.nansum(jnp.nansum((roots_State.roots.imag[0:-1]+roots_State.roots.imag[1:])*
                                  (roots_State.roots.real[0:-1]-roots_State.roots.real[1:])*roots_State.parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(roots_State,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    add_mag_no_diff_num = (jnp.abs(mag-maglast)<1/2*epsilon).sum()
    # check the change of the mag, if the mag is not changed at least 2 iteration, we stop the loop
    error_hist+=buried_error
    mag_State = Error_State(mag,
                            add_mag_no_diff_num+mag_State_last.mag_no_diff,
                            add_outloop+mag_State_last.outloop,
                            error_hist,epsilon,epsilon_rel)
    return mag_State
