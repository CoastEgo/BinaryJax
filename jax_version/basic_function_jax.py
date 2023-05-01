import numpy as np
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from scipy.optimize import linear_sum_assignment
import jax
from jax import lax
from jax import custom_jvp
from jax import jvp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
idx_all=jnp.linspace(0,4,5,dtype=int)
def fz0(z,m1,m2,s):
    return -m1/(z-s)-m2/z
def fz1(z,m1,m2,s):
    return m1/(z-s)**2+m2/z**2
def fz2(z,m1,m2,s):
    return -2*m1/(z-s)**3-2*m2/z**3
def fz3(z,m1,m2,s):
    return 6*m1/(z-s)**4+6*m2/z**4
def J(z,m1,m2,s):
    return 1-fz1(z,m1,m2,s)*jnp.conj(fz1(z,m1,m2,s))
def Quadrupole_test(rho,s,q,zeta,z,zG,tol):
    m1=1/(1+q)
    m2=q/(1+q)
    cQ=6;cG=2;cP=2
    ####Quadrupole test
    miu_Q=jnp.abs(-2*jnp.real(3*jnp.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2-(3-3*J(z,m1,m2,s)+J(z,m1,m2,s)**2/2)*jnp.abs(fz2(z,m1,m2,s))**2+J(z,m1,m2,s)*jnp.conj(fz1(z,m1,m2,s))**2*fz3(z,m1,m2,s))/(J(z,m1,m2,s)**5))
    miu_C=jnp.abs(6*jnp.imag(3*jnp.conj(fz1(z,m1,m2,s))**3*fz2(z,m1,m2,s)**2)/(J(z,m1,m2,s)**5))
    cond1=jnp.nansum(miu_Q+miu_C,axis=1)*cQ*(rho**2+1e-4*tol)<tol
    ####ghost image test
    zwave=jnp.conj(zeta[:,jnp.newaxis])-fz0(zG,m1,m2,s)
    J_wave=1-fz1(zG,m1,m2,s)*fz1(zwave,m1,m2,s)
    miu_G=1/2*jnp.abs(J(zG,m1,m2,s)*J_wave**2/(J_wave*fz2(jnp.conj(zG),m1,m2,s)*fz1(zG,m1,m2,s)-jnp.conj(J_wave)*fz2(zG,m1,m2,s)*fz1(jnp.conj(zG),m1,m2,s)*fz1(zwave,m1,m2,s)))
    cond2=~((cG*(rho+1e-3)>miu_G).any(axis=1))#any更加宽松，因为ghost roots应该是同时消失的，理论上是没问题的
    #####planet test
    cond3=(q>1e-2)|(jnp.abs(zeta+1/s)**2>cP*(rho**2+9*q/s**2))|(rho*rho*s*s<q)
    return cond1&cond2&cond3,jnp.nansum(jnp.abs(1/J(z,m1,m2,s)),axis=1)
@jax.jit
def get_poly_coff(zeta_l,s,m2):
    zeta_conj=jnp.conj(zeta_l)
    c0=s**2*zeta_l*m2**2
    c1=-s*m2*(2*zeta_l+s*(-1+s*zeta_l-2*zeta_l*zeta_conj+m2))
    c2=zeta_l-s**3*zeta_l*zeta_conj+s*(-1+m2-2*zeta_conj*zeta_l*(1+m2))+s**2*(zeta_conj-2*zeta_conj*m2+zeta_l*(1+zeta_conj**2+m2))
    c3=s**3*zeta_conj+2*zeta_l*zeta_conj+s**2*(-1+2*zeta_conj*zeta_l-zeta_conj**2+m2)-s*(zeta_l+2*zeta_l*zeta_conj**2-2*zeta_conj*m2)
    c4=zeta_conj*(-1+2*s*zeta_conj+zeta_conj*zeta_l)-s*(-1+2*s*zeta_conj+zeta_conj*zeta_l+m2)
    c5=(s-zeta_conj)*zeta_conj
    coff=jnp.stack((c5,c4,c3,c2,c1,c0),axis=1)
    return coff
def verify(zeta_l,z_l,s,m1,m2):#verify whether the root is right
    return  jnp.abs(z_l-m1/(jnp.conj(z_l)-s)-m2/jnp.conj(z_l)-zeta_l)
def get_parity(z,s,m1,m2):#get the parity of roots
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.sign((1-jnp.abs(de_conjzeta_z1)**2))
def get_parity_error(z,s,m1,m2):
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.abs((1-jnp.abs(de_conjzeta_z1)**2))
@jax.jit # 定义函数以进行矢量化
def loop_body(k, coff):
    return jnp.roots(coff[k],strip_zeros=False)
def get_roots(sample_n, coff):
    # 使用 vmap 进行矢量化，并指定输入参数的轴数
    roots = jax.vmap(loop_body, in_axes=(0, None))(jnp.arange(sample_n), coff)
    return roots
def dot_product(a,b):
    return np.real(a)*np.real(b)+np.imag(a)*np.imag(b)
@custom_jvp
@jax.jit
def find_nearest(array1, parity1, array2, parity2):#线性分配问题
    cost=jnp.abs(array2-array1[:,None])+jnp.abs(parity2-parity1[:,None])*5#系数可以指定防止出现错误，系数越大鲁棒性越好，但是速度会变慢些
    cost=jnp.where(jnp.isnan(cost),100,cost)
    col_idx = hcb.call(lsa,cost,result_shape=jax.ShapeDtypeStruct(cost[:,-1].shape, jnp.int64))
    return col_idx
@find_nearest.defjvp
def find_nearest_jvp(primals, tangents):
    a1,p1,a2,p2=primals
    da1,dp1,da2,dp2=tangents
    primals_out=find_nearest(a1,p1,a2,p2)
    idx=primals_out[2]
    idx_bcast = jnp.broadcast_to(idx, da2.shape)
    da2_out = jnp.take(da2, idx_bcast)
    dp2_out = jnp.take(dp2, idx_bcast)
    return primals_out,jnp.zeros_like(primals_out)
def lsa(cost):
    row_ind, col_idx=linear_sum_assignment(cost)
    return col_idx
@jax.jit
def sort_body1(values,k):
    roots, parity = values
    sort_indices = find_nearest(roots[k - 1, :], parity[k - 1, :], roots[k, :], parity[k, :])
    roots = roots.at[k, :].set(roots[k, sort_indices])
    parity = parity.at[k, :].set(parity[k, sort_indices])
    return (roots,parity),k
def get_sorted_roots(index, roots, parity):
    carry,_=lax.scan(sort_body1,(roots, parity),index)
    return carry
@jax.jit
def sort_body2(carry,i):
    roots,parity=carry
    sort_indices=find_nearest(roots[i],parity[i],roots[i+1],parity[i+1])
    cond = jnp.tile(jnp.arange(roots.shape[0])[:, None], (1, roots.shape[1])) < i+1
    roots=jnp.where(cond,roots,roots[:,sort_indices])
    parity=jnp.where(cond,parity,parity[:,sort_indices])
    return (roots,parity),i