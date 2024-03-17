import numpy as jnp
import jax.numpy as jnp
import jax
from jax import lax
from .error_estimator import *
from .solution import *
from .basic_function_jax import Quadrupole_test
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

@jax.jit
def model(t_0,u_0,t_E,rho,q,s,alpha_deg,times,retol=0.001):
    # Here the parameterization is consistent with Mulensmodel and VBBinaryLensing
    # But the alpha is 180 degree different from VBBinaryLensing
    ### initialize parameters
    alpha_rad=alpha_deg*2*jnp.pi/360
    times=(times-t_0)/t_E
    trajectory_n=times.shape[0]
    m1=1/(1+q)
    m2=q/(1+q)
    trajectory_l=get_trajectory_l(s,q,alpha_rad,u_0,times)
    ###四极测试
    zeta_l=trajectory_l[:,None]
    coff=get_poly_coff(zeta_l,s,m2)
    z_l=get_roots(trajectory_n,coff)
    error=verify(zeta_l,z_l,s,m1,m2)
    cond=error<1e-6
    #index=jnp.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5))[0]
    '''if index.size!=0:
        sortidx=jnp.argsort(error[index],axis=1)
        cond=cond.at[index].set(False)
        cond=cond.at[index,sortidx[:,0:3]].set(True)'''
    ###########
    index=jnp.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5),size=5,fill_value=-1)[0]
    def ambigious_deal(carry):
        error,index,cond=carry
        sortidx=jnp.argsort(error[index],axis=1)
        cond=cond.at[index].set(False)
        cond=cond.at[index[:,None],sortidx[:,0:3]].set(True)
        return cond
    cond=lax.cond((index!=-1).any(),ambigious_deal,lambda x : x[-1],(error,index,cond)) #某些情况下出现的根不是5或者3个#'''
    #############
    z=jnp.where(cond,z_l,jnp.nan)
    zG=jnp.where(cond,jnp.nan,z_l)
    cond,mag=Quadrupole_test(rho,s,q,zeta_l,z,zG)
    #cond=cond.at[:].set(False)
    carry,_=lax.scan(contour_scan,(mag,trajectory_l,retol,retol,rho,s,q,m1,m2,
                                    jnp.array([0]),cond,False),jnp.arange(trajectory_n))
    mag,trajectory_l,tol,retol,rho,s,q,m1,m2,sample_n,cond,outlop=carry#'''
    #print(sample_n)
    '''for i in range(len(cond)):
        if cond[i]==False:
            result=contour_integrate(rho,s,q,m1,m2,trajectory_l[i],retol,epsilon_rel=retol)
            (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
            Is_create,_,rho,s,q,m1,m2,epsilon,epsilon_rel,mag_now,maglast,outloop)=result
            print(sample_n)
            print(mag_now)
            print(maglast)
            print(jnp.sum(error_hist))
            print(error_hist)
            print((error_hist/jnp.abs(mag_now)>retol/2/jnp.sqrt(sample_n)).any())
            mag=mag.at[i].set(mag_now[0])#jax perfetto代码测试#'''
    return mag
def to_centroid(s,q,x):#change coordinate system to cetorid
    delta_x=s/(1+q)
    return -(jnp.conj(x)-delta_x)
def to_lowmass(s,q,x):#change coordinaate system to lowmass
    delta_x=s/(1+q)
    return -jnp.conj(x)+delta_x
@jax.jit
def get_trajectory_l(s,q,alpha_rad,u_0,times):
    alpha=alpha_rad
    b=u_0
    trajectory_l=to_lowmass(s,q,times*jnp.cos(alpha)-b*jnp.sin(alpha)+1j*(b*jnp.cos(alpha)+times*jnp.sin(alpha)))
    return trajectory_l
@jax.jit
def contour_scan(carry,i):
    mag_all,trajectory_l,retol,retol,rho,s,q,m1,m2,sample_n,cond,outlop=carry
    @jax.jit
    def clear_trash(carry):
        (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)=carry
        ## clear trash value in the last five elements
        theta=theta.at[-3:].set(jnp.nan)
        error_hist=error_hist.at[-3:].set(0.)
        roots=roots.at[-3:].set(jnp.nan)
        parity=parity.at[-3:].set(jnp.nan)
        ghost_roots_dis=ghost_roots_dis.at[-3:].set(jnp.nan)
        buried_error=buried_error.at[-3:].set(0.)
        sort_flag=sort_flag.at[-3:].set(True)
        Is_create=Is_create.at[-3:].set(0)
        carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)
        return carry
    @partial(jax.jit,static_argnums=(-1,))
    def reshape_fun(carry,arraylength):
        (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,maglast,outloop)=carry
        ## reshape the array and fill the new array with nan
        pad_list = [theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,Is_create]
        pad_value = [jnp.nan,0.,jnp.nan,jnp.nan,jnp.nan,0.,True,0]
        padded_list =jax.tree_map(lambda x,y: jnp.pad(x,((0,arraylength),(0,0)),'constant',constant_values=y),pad_list,pad_value)

        theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,Is_create=padded_list
        carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,epsilon,epsilon_rel,mag,maglast,outloop)
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
        return resultnew,Max_array_length
    @jax.jit
    def false_fun(carry):
        mag_all,trajectory_l,retol,retol,rho,s,q,m1,m2,sample_n,cond,outlop=carry
        ## JIT compile operation needs shape of the array to be determined.
        ## But for optimial sampling, It is hard to know the array length before the code runs so we need to assign large enough array length
        ## which will cause the waste of memory and time. 
        ## To solve this problem, here we use heriachical array length adding method to add array length gradually,
        ## the problem is we should fine tuen the array length added in different layers to get the optimal performance which depends on the tolerance and parameter.
        ## current is 60 + 80 + 150 = 290

        ## first add
        result=contour_integrate(rho,s,q,m1,m2,trajectory_l[i],retol,epsilon_rel=retol)
        result,resultlast=result
        Max_array_length=60### the array length in different layers need to be determined
        ## if max array length is reached, try to add array length
        ### reshape the array and fill the new array with nan

        ## second add
        sample_n=result[0]
        add_length=80##80
        resultlast=reshape_fun(resultlast,add_length)
        result=reshape_fun(result,add_length)

        result,Max_array_length=lax.cond((sample_n<Max_array_length-5)[0],lambda x:(x[0],x[-1]),secondary_contour,(result,resultlast,add_length,Max_array_length))
        
        ## third add
        sample_n=result[0]
        add_length=150
        resultlast=reshape_fun(resultlast,add_length)
        result=reshape_fun(result,add_length)

        result,Max_array_length=lax.cond((sample_n<Max_array_length-5)[0],lambda x:(x[0],x[-1]),secondary_contour,(result,resultlast,add_length,Max_array_length))

        (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,_,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)=result
        mag=lax.cond((sample_n<Max_array_length-5)[0],lambda x:x[0],lambda x:x[1],(mag,maglast))
        mag_all=mag_all.at[i].set(mag[0])
        return (mag_all,sample_n,outlop)
    mag_all,sample_n,outlop=lax.cond(cond[i].all(),lambda x:(x[0],x[-3],x[-1]),false_fun,carry)
    return (mag_all,trajectory_l,retol,retol,rho,s,q,m1,m2,sample_n,cond,outlop),sample_n
@partial(jax.jit,static_argnums=(-1,))
def contour_integrate(rho,s,q,m1,m2,trajectory_l,epsilon,epsilon_rel=0,inite=30,n_ite=60):
    ###初始化
    sample_n=jnp.array([inite])
    theta=jnp.where(jnp.arange(n_ite)<inite,jnp.resize(jnp.linspace(0,2*jnp.pi,inite),n_ite),jnp.nan)[:,None]#shape(500,1)
    error_hist=jnp.ones(n_ite)
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    coff=get_poly_coff(zeta_l,s,m2)
    roots,parity,ghost_roots_dis,outloop,coff,zeta_l,theta=get_real_roots(coff,zeta_l,theta,s,m1,m2)
    buried_error=get_buried_error(ghost_roots_dis,sample_n)
    sort_flag=jnp.where(jnp.arange(n_ite)<inite,False,True)[:,None]#是否需要排序
    ### no need to sort first idx
    sort_flag=sort_flag.at[0].set(True)
    roots,parity,sort_flag=get_sorted_roots(roots,parity,sort_flag)
    Is_create=find_create_points(roots,sample_n)
    #####计算第一次的误差，放大率
    maglast=jnp.array([1.])
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
            Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)
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
def cond_fun(carry):
    carry,carrylast=carry
    ## function to judge whether to continue the loop use relative error
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)=carry
    Max_array_length=jnp.shape(theta)[0]
    mini_interval=jnp.nanmin(jnp.abs(jnp.diff(theta,axis=0)))
    abs_mag_cond=(jnp.nansum(error_hist)>epsilon_rel)
    abs_mag_cond2=(error_hist>epsilon_rel/jnp.sqrt(sample_n)).any()
    rel_mag_cond=(error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n)).any()
    #rel_mag_cond=(jnp.nansum(error_hist)/jnp.abs(mag)>epsilon_rel)[0]
    relmag_diff_cond=(jnp.abs((mag-maglast)/maglast)>1/2*epsilon_rel)[0]
    mag_diff_cond=(jnp.abs(mag-maglast)>1/2*epsilon_rel)[0]

    ## switch the different stopping condition: absolute error or relative error
    ## to modify the stopping condition, you will also need to modify the add points method in the while_body_fun

    loop= (rel_mag_cond& (mini_interval>1e-14)& (~outloop)& abs_mag_cond & (mag_diff_cond|(sample_n<Max_array_length/2)[0]) & (sample_n<Max_array_length-5)[0])
    
    #loop= (abs_mag_cond2&(mini_interval>1e-14)& (~outloop)& abs_mag_cond & (mag_diff_cond|(sample_n<Max_array_length/2)[0]) & (sample_n<Max_array_length-5)[0])
    return loop
@jax.jit
def while_body_fun(carry):
    carry,carrylast=carry
    carrylast=carry
    ## function to add points, calculate the error and mag
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)=carry
    add_max=4
    Max_array_length=jnp.shape(theta)[0]
    ######迭代加点
    #一次一个区间加点：
    '''idx=jnp.nanargmax(error_hist)#单区间多点采样
    add_number=jnp.ceil((error_hist[idx]/jnp.abs(mag)/epsilon_rel*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    add_number=jax.lax.min(add_number,add_max)
    theta_diff = (theta[idx] - theta[idx-1]) / (add_number[0]+1)
    add_theta=jnp.arange(1,add_max+1)[:,None]*theta_diff+theta[idx-1]
    add_theta=jnp.where((jnp.arange(add_max)<add_number)[:,None],add_theta,jnp.nan)'''
    #一次多个区间加点:
    
    ### absolute error adding mode

    #idx=jnp.where(error_hist>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length/5),fill_value=0)[0]
    #add_number=jnp.ceil((error_hist[idx]/epsilon_rel*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    
    ## relative error adding mode

    idx=jnp.where(error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n),size=int(Max_array_length/5),fill_value=0)[0]
    add_number=jnp.ceil((error_hist[idx]/jnp.abs(mag)/epsilon_rel*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    
    add_number=jnp.where((idx==0)[:,None],0,add_number)
    add_number=jnp.where(add_number>add_max,add_max,add_number)
    '''carry=(theta,idx,add_number,jnp.full(theta.shape,jnp.nan))
    for i in range(idx.shape[0]):
        carry,k=theta_encode(carry,i)'''
    carry,_=lax.scan(theta_encode,(theta,idx,add_number,
                               jnp.full(theta.shape,jnp.nan)),jnp.arange(idx.shape[0]))
    add_theta=carry[-1]
    ####
    add_zeta_l=get_zeta_l(rho,trajectory_l,add_theta)
    add_coff=get_poly_coff(add_zeta_l,s,m2)
    sample_n+=jnp.sum(add_number)
    theta,ghost_roots_dis,buried_error,sort_flag,roots,parity,Is_create,outloop=add_points(
        idx,add_zeta_l,add_coff,add_theta,roots,parity,theta,ghost_roots_dis,sort_flag,s,m1,m2,sample_n,add_number)
    ####计算误差
    maglast=mag
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,maglast,outloop)
    return (carry,carrylast)
