import numpy as jnp
import jax.numpy as jnp
import jax
from jax import lax
from error_estimator import *
from solution import *
from basic_function_jax import Quadrupole_test
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
class model():#initialize parameter
    def __init__(self,par):
        self.t_0=par['t_0']
        self.u_0=par['u_0']
        self.t_E=par['t_E']
        self.rho=par['rho']
        self.q=par['q']
        self.s=par['s']
        self.alpha_rad=par['alpha_deg']*2*jnp.pi/360
        self.times=(par['times']-self.t_0)/self.t_E
        self.trajectory_n=len(self.times)
        self.m1=1/(1+self.q)
        self.m2=self.q/(1+self.q)
        self.trajectory_l=self.get_trajectory_l()
    def to_centroid(self,x):#change coordinate system to cetorid
        delta_x=self.s/(1+self.q)
        return -(jnp.conj(x)-delta_x)
    def to_lowmass(self,x):#change coordinaate system to lowmass
        delta_x=self.s/(1+self.q)
        return -jnp.conj(x)+delta_x
    def get_trajectory_l(self):
        alpha=self.alpha_rad
        b=self.u_0
        trajectory_l=self.to_lowmass(self.times*jnp.cos(alpha)-b*jnp.sin(alpha)+1j*(b*jnp.cos(alpha)+self.times*jnp.sin(alpha)))
        return trajectory_l
    def get_magnifaction2(self,tol,retol=0):
        trajectory_l=self.trajectory_l
        trajectory_n=self.trajectory_n
        zeta_l=trajectory_l[:,None]
        coff=get_poly_coff(zeta_l,self.s,self.m2)
        z_l=get_roots(trajectory_n,coff)
        error=verify(zeta_l,z_l,self.s,self.m1,self.m2)
        cond=error<1e-6
        index=jnp.where((cond.sum(axis=1)!=3) & (cond.sum(axis=1)!=5))[0]
        if index.size!=0:
            sortidx=jnp.argsort(error[index],axis=1)
            cond.at[index].set(False)
            cond.at[index,sortidx[0:3]].set(True)
        z=jnp.where(cond,z_l,jnp.nan)
        zG=jnp.where(cond,jnp.nan,z_l)
        cond,mag=Quadrupole_test(self.rho,self.s,self.q,zeta_l,z,zG,tol)
        idx=jnp.where(~cond)[0]
        carry,_=lax.scan(contour_scan,(mag,trajectory_l,tol,retol,self.rho,self.s,self.q,self.m1,self.m2),idx)
        mag,trajectory_l,tol,retol,rho,s,q,m1,m2=carry
        return mag
def contour_scan(carry,i):
    mag,trajectory_l,tol,retol,rho,s,q,m1,m2=carry
    temp_mag=contour_integrate(rho,s,q,m1,m2,trajectory_l[i],tol,epsilon_rel=retol)
    mag=mag.at[i].set(temp_mag[0])
    return (mag,trajectory_l,tol,retol,rho,s,q,m1,m2),i
def contour_integrate(rho,s,q,m1,m2,trajectory_l,epsilon,epsilon_rel=0,inite=5,n_ite=500):
    ###初始化
    sample_n=jnp.array([inite])
    theta=jnp.where(jnp.arange(n_ite)<inite,jnp.resize(jnp.linspace(0,2*jnp.pi,inite),n_ite),jnp.nan)[:,None]#shape(500,1)
    error_hist=jnp.ones(n_ite)
    zeta_l=get_zeta_l(rho,trajectory_l,theta)
    coff=get_poly_coff(zeta_l,s,m2)
    roots,parity,ghost_roots_dis,outloop,coff,zeta_l,theta=get_real_roots(coff,zeta_l,theta,s,m1,m2)
    buried_error=get_buried_error(ghost_roots_dis)
    sort_flag=jnp.where(jnp.arange(n_ite)<inite,False,True)[:,None]#是否需要排序
    roots,parity,sort_flag=get_sorted_roots(roots,parity,sort_flag)
    Is_create=find_create_points(roots,sample_n)
    #####计算第一次的误差，放大率
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
            Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,outloop)
    result=lax.while_loop(cond_fun,while_body_fun,carry)
    return result[-2]
def cond_fun(carry):
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,outloop)=carry
    mini_interval=jnp.nanmin(jnp.abs(jnp.diff(theta,axis=0)))
    loop=((error_hist>epsilon/jnp.sqrt(sample_n)).any() & (error_hist/jnp.abs(mag)>epsilon_rel/jnp.sqrt(sample_n)).any() & (mini_interval>1e-14) & (~outloop))
    return loop
def while_body_fun(carry):
    (sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,outloop)=carry
    idx=jnp.nanargmax(error_hist)#单区间多点采样
    #idx=jnp.where(error_hist>epsilon/jnp.sqrt(sample_n))[0]#不满足要求的点
    ######迭代加点
    add_number=jnp.ceil((error_hist[idx]/epsilon*jnp.sqrt(sample_n))**0.2).astype(int)#至少要插入一个点，不包括相同的第一个
    theta_diff = (theta[idx] - theta[idx-1]) / (add_number[0]+1)
    add_theta=jnp.arange(1,theta.shape[0]+1)[:,None]*theta_diff+theta[idx-1]
    add_theta=jnp.where((jnp.arange(theta.shape[0])<add_number)[:,None],add_theta,jnp.nan)
    #add_theta=[jnp.linspace(theta[idx[i]-1],theta[idx[i]],add_number[i],endpoint=False)[1:] for i in range(jnp.shape(idx)[0])]
    #idx = jnp.repeat(idx, add_number-1) # create an index array with the same length as add_item
    #add_theta = jnp.concatenate(add_theta) # concatenate the list of arrays into a 1-D array
    add_zeta_l=get_zeta_l(rho,trajectory_l,add_theta)
    add_coff=get_poly_coff(add_zeta_l,s,m2)
    sample_n+=add_number
    theta,ghost_roots_dis,buried_error,sort_flag,roots,parity,Is_create,outloop=add_points(
        idx,add_zeta_l,add_coff,add_theta,roots,parity,theta,ghost_roots_dis,sort_flag,s,m1,m2,sample_n,add_number)
    ####计算误差
    mag=1/2*jnp.nansum(jnp.nansum((roots.imag[0:-1]+roots.imag[1:])*(roots.real[0:-1]-roots.real[1:])*parity[0:-1],axis=0))
    error_hist,magc,parab=error_sum(Is_create,roots,parity,theta,rho,q,s)
    mag=(mag+magc+parab)/(jnp.pi*rho**2)
    error_hist+=buried_error
    carry=(sample_n,theta,error_hist,roots,parity,ghost_roots_dis,buried_error,sort_flag,
        Is_create,trajectory_l,rho,s,q,m1,m2,epsilon,epsilon_rel,mag,outloop)
    return carry
