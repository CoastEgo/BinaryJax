import numpy as np
import jax.numpy as jnp
from .polynomial_solver import halfanalytical,zroots,implict_zroots
import jax
from functools import partial
from jax import lax
from jax import custom_jvp
from .linear_sum_assignment_jax import solve
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
@jax.jit
def fz0(z,m1,m2,s):
    return -m1/(z-s)-m2/z
@jax.jit
def fz1(z,m1,m2,s):
    return m1/(z-s)**2+m2/z**2
@jax.jit
def fz2(z,m1,m2,s):
    return -2*m1/(z-s)**3-2*m2/z**3
@jax.jit
def fz3(z,m1,m2,s):
    return 6*m1/(z-s)**4+6*m2/z**4
@jax.jit
def J(z,m1,m2,s):
    return 1-fz1(z,m1,m2,s)*jnp.conj(fz1(z,m1,m2,s))
@jax.jit
def Quadrupole_test(rho,s,q,zeta,z,zG,tol):
    m1=1/(1+q)
    m2=q/(1+q)
    cQ=6;cG=2;cP=2
    ####Quadrupole test
    miu_Q=jnp.abs(-2*jnp.real(3*jnp.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2-(3-3*J(z,m1,m2,s)+J(z,m1,m2,s)**2/2)*jnp.abs(fz2(z,m1,m2,s))**2+J(z,m1,m2,s)*jnp.conj(fz1(z,m1,m2,s))**2*fz3(z,m1,m2,s))/(J(z,m1,m2,s)**5))
    miu_C=jnp.abs(6*jnp.imag(3*jnp.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2)/(J(z,m1,m2,s)**5))
    mag=jnp.nansum(jnp.abs(1/J(z,m1,m2,s)),axis=1)
    tol*=mag
    cond1=jnp.nansum(miu_Q+miu_C,axis=1)*cQ*(rho**2+1e-4*tol)<tol
    ####ghost image test
    zwave=jnp.conj(zeta)-fz0(zG,m1,m2,s)
    J_wave=1-fz1(zG,m1,m2,s)*fz1(zwave,m1,m2,s)
    miu_G=1/2*jnp.abs(J(zG,m1,m2,s)*J_wave**2/(J_wave*fz2(jnp.conj(zG),m1,m2,s)*fz1(zG,m1,m2,s)-jnp.conj(J_wave)*fz2(zG,m1,m2,s)*fz1(jnp.conj(zG),m1,m2,s)*fz1(zwave,m1,m2,s)))
    cond2=~((cG*(rho+1e-3)>miu_G).any(axis=1))#any更加宽松，因为ghost roots应该是同时消失的，理论上是没问题的
    #####planet test
    cond3=((q>1e-2)|(jnp.abs(zeta+1/s)**2>cP*(rho**2+9*q/s**2))|(rho*rho*s*s<q))[:,0]
    return cond1&cond2&cond3,mag
@jax.jit
def get_poly_coff(zeta_l,s,m2):
    zeta_conj=jnp.conj(zeta_l)
    c0=s**2*zeta_l*m2**2
    c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
    c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
    c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
    c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
    c5=(s-zeta_conj)*zeta_conj
    coff=jnp.concatenate((c5,c4,c3,c2,c1,c0),axis=1)
    return coff
@jax.jit
def get_zeta_l(rho,trajectory_centroid_l,theta):#获得等高线采样的zeta
    rel_centroid=rho*jnp.cos(theta)+1j*rho*jnp.sin(theta)
    zeta_l=trajectory_centroid_l+rel_centroid
    return zeta_l
@jax.jit
def verify(zeta_l,z_l,s,m1,m2):#verify whether the root is right
    return  jnp.abs(z_l-m1/(jnp.conj(z_l)-s)-m2/jnp.conj(z_l)-zeta_l)
@jax.jit
def get_parity(z,s,m1,m2):#get the parity of roots
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.sign((1-jnp.abs(de_conjzeta_z1)**2))
@jax.jit
def get_parity_error(z,s,m1,m2):
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.abs((1-jnp.abs(de_conjzeta_z1)**2))
'''@jax.jit # 自动矢量化
def loop_body(k, coff):
    #roots=jnp.roots(coff[k],strip_zeros=False)
    roots=halfanalytical(coff[k])
    return roots#自动矢量化，但是有浪费'''
@jax.jit # 定义函数以进行矢量化
def loop_body(carry,k):#采用判断来减少浪费
    coff,roots=carry
    @jax.jit
    def False_fun(carry):
        coff,roots,k=carry
        roots=roots.at[k].set(jnp.roots(coff,strip_zeros=False))
        #roots=roots.at[k].set(halfanalytical(coff))
        #roots=roots.at[k].set(implict_zroots(coff))
        return roots
    roots=lax.cond((coff[k]==0).all(),lambda x:x[1],False_fun,(coff[k],roots,k))
    return (coff,roots),k#'''
@partial(jax.jit,static_argnums=0)
def get_roots(sample_n, coff):
    # 使用 vmap 进行矢量化，并指定输入参数的轴数
    #roots = jax.vmap(loop_body, in_axes=(0, None))(jnp.arange(sample_n), coff)
    carry,_=lax.scan(loop_body,(coff,jnp.zeros((coff.shape[0],5),dtype=jnp.complex128)),jnp.arange(sample_n))#scan循环，但是没有浪费
    coff,roots=carry
    return roots
@jax.jit
def dot_product(a,b):
    return np.real(a)*np.real(b)+np.imag(a)*np.imag(b)
@custom_jvp
@jax.jit
def find_nearest(array1, parity1, array2, parity2):#线性分配问题
    cost=jnp.abs(array2-array1[:,None])+jnp.abs(parity2-parity1[:,None])*5#系数可以指定防止出现错误，系数越大鲁棒性越好，但是速度会变慢些
    cost=jnp.where(jnp.isnan(cost),100,cost)
    row_ind, col_idx=solve(cost)
    return col_idx
@find_nearest.defjvp
def find_nearest_jvp(primals, tangents):
    a1,p1,a2,p2=primals
    #da1,dp1,da2,dp2=tangents
    primals_out=find_nearest(a1,p1,a2,p2)
    #idx=primals_out[2]
    #idx_bcast = jnp.broadcast_to(idx, da2.shape)
    #da2_out = jnp.take(da2, idx_bcast)
    #dp2_out = jnp.take(dp2, idx_bcast)
    return primals_out,jnp.zeros_like(primals_out)
'''@jax.jit
def custom_insert(array,idx,add_array,add_number):
    ite=jnp.arange(array.shape[0])
    mask = ite < idx
    array=jnp.where(mask[:,None],array,jnp.roll(array,add_number,axis=0))
    mask2=(ite >=idx)&(ite<idx+add_number)
    add_array=jnp.resize(add_array,array.shape)
    add_array=jnp.roll(add_array,idx,axis=0)
    array=jnp.where(mask2[:,None],add_array,array)
    return array'''
@jax.jit
def insert_body(carry,k):
    array,add_array,idx,add_number=carry
    ite=jnp.arange(array.shape[0])
    mask = ite < idx[k]
    array=jnp.where(mask[:,None],array,jnp.roll(array,add_number[k],axis=0))
    mask2=(ite >=idx[k])&(ite<idx[k]+add_number[k])
    add_array=jnp.roll(add_array,idx[k],axis=0)
    array=jnp.where(mask2[:,None],add_array,array)
    add_array=jnp.roll(add_array,-1*add_number[k]-idx[k],axis=0)
    idx+=add_number[k]
    return (array,add_array,idx,add_number),k
@jax.jit
def custom_insert(array,idx,add_array,add_number):
    carry,_=lax.scan(insert_body,(array,add_array,idx,add_number),jnp.arange(idx.shape[0]))
    array,add_array,idx,add_number=carry
    return array
@jax.jit
def theta_encode(carry,k):
    (theta,idx,add_number,add_theta_encode)=carry
    add_max=theta.shape[0]
    theta_diff = (theta[idx[k]] - theta[idx[k]-1]) / (add_number[k]+1)
    add_theta=jnp.arange(1,add_max+1)[:,None]*theta_diff+theta[idx[k]-1]
    add_theta=jnp.where((jnp.arange(add_max)<add_number[k])[:,None],add_theta,jnp.nan)
    carry2,_=insert_body((add_theta_encode,add_theta,jnp.where(jnp.isnan(add_theta_encode),size=1)[0],add_number[k][None]),0)
    add_theta_encode=carry2[0]
    return (theta,idx,add_number,add_theta_encode),k
@jax.jit
def delete_body(carry, k):
    array, ite2 = carry
    mask = ite2 < k
    array = jnp.where(mask[:,None], array, jnp.roll(array, -1,axis=0))
    return (array, ite2 + 1), k
@jax.jit
def custom_delete(array, idx):
    ite = jnp.arange(array.shape[0])
    carry, _ = lax.scan(delete_body, (array, ite), idx)
    array, _ = carry
    array = jnp.where((ite < ite.size - (idx<array.shape[0]).sum())[:,None], array, jnp.nan)
    return array
