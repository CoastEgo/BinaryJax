import numpy as np
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
from scipy.optimize import linear_sum_assignment
import jax
from jax import lax
from functools import partial
jax.config.update("jax_enable_x64", True)
idx_all=jnp.linspace(0,4,5,dtype=int)
@partial(jax.jit,static_argnums=(1,2))
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
@partial(jax.jit,static_argnums=(2,3,4))
def verify(zeta_l,z_l,s,m1,m2):#verify whether the root is right
    return  jnp.abs(z_l-m1/(jnp.conj(z_l)-s)-m2/jnp.conj(z_l)-zeta_l)
@partial(jax.jit,static_argnums=(1,2,3))
def get_parity(z,s,m1,m2):#get the parity of roots
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.sign((1-jnp.abs(de_conjzeta_z1)**2))
@partial(jax.jit,static_argnums=(1,2,3))
def get_parity_error(z,s,m1,m2):
    de_conjzeta_z1=m1/(jnp.conj(z)-s)**2+m2/jnp.conj(z)**2
    return jnp.abs((1-jnp.abs(de_conjzeta_z1)**2))
def get_roots(sample_n, coff):
    # 定义函数以进行矢量化
    def loop_body(k, coff):
        return jnp.roots(coff[k, :], strip_zeros=False)
    # 使用 vmap 进行矢量化，并指定输入参数的轴数
    roots = jax.vmap(loop_body, in_axes=(0, None))(jnp.arange(sample_n), coff)
    return roots
@jax.jit
def dot_product(a,b):
    return jnp.real(a)*jnp.real(b)+jnp.imag(a)*jnp.imag(b)
@jax.jit
def find_nearest(array1, parity1, array2, parity2):#线性分配问题
    cost=jnp.abs(array2-array1[:,None])+jnp.abs(parity2-parity1[:,None])*5#系数可以指定防止出现错误，系数越大鲁棒性越好，但是速度会变慢些
    cost=jnp.where(jnp.isnan(cost),100,cost)
    col_idx = hcb.call(lsa,cost,result_shape=jax.ShapeDtypeStruct(cost[:,-1].shape, jnp.int64))
    return col_idx
def lsa(cost):
    row_ind, col_idx=linear_sum_assignment(cost)
    return col_idx
def get_sorted_roots(sample_n, roots, parity):
    def body_fun(k, values):
        roots, parity = values
        sort_indices = find_nearest(roots[k - 1, :], parity[k - 1, :], roots[k, :], parity[k, :])
        roots = roots.at[k, :].set(roots[k, sort_indices])
        parity = parity.at[k, :].set(parity[k, sort_indices])
        return (roots,parity)
    init_val = (roots, parity)
    roots,parity = lax.fori_loop(1, sample_n, body_fun, init_val)
    return roots, parity
def search(m_map,n_map,roots,parity,fir_val,Is_create):#图像匹配算法
    m=m_map[-1]
    n=n_map[-1]
    sample_n=jnp.shape(roots)[0]
    if (m>=sample_n)|(m<0):#循环列表
        m=m%(sample_n)
    m_next=(m-int(parity[m,n]))%sample_n
    nextisnan=(jnp.isnan(roots[m_next,n]))
    if len(m_map)!=1:
    #如果下一个已经闭合
        if jnp.isclose(roots[m,n],fir_val,rtol=1e-7):
            parity=parity.at[m,n].set(0)
            roots=roots.at[m,n].set(jnp.nan)
            return m_map,n_map,roots,parity  
    if ((m==sample_n-1)&(m_next==0))|((m==0)&(m_next==sample_n-1)):#处理首尾相连的问题(0与2pi)
        m_next=m_next%sample_n
        try:
            transit=jnp.where(jnp.isclose(roots[m_next,:],roots[m,n],rtol=1e-6))[0][0]
            roots=roots.at[m,n].set(jnp.nan)
            parity=parity.at[m,n].set(0)
            m_map+=[m_next];n_map+=[transit]
            m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
            return m_map_res,n_map_res,temp_roots,temp_parity
        except IndexError:
            parity=parity.at[m,n].set(0)
            roots=roots.at[m,n].set(jnp.nan)
            return m_map,n_map,roots,parity    
    #如果下一个不是nan，继续往下走
    if (~nextisnan):
        par=-int(parity[m,n])
        try:
            critial_m=jnp.where(jnp.isnan(roots[m::par,n]))[0][0]-1
            real_m=m+par*critial_m
        except IndexError:
            real_m=int(-1/2*(par+1))%sample_n
        roots=roots.at[m:real_m:par,n].set(jnp.nan)
        parity=parity.at[m:real_m:par,n].set(0)
        m_map+=[i for i in range(m+par,real_m+par,par)]
        n_map+=[n]*(abs(real_m-m))
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果下一个是nan并且当前还有别的列，就转换列,换列不能发生在最后一行因为 2*pi=0此时根个数相同不存在create destruct
    elif (nextisnan & (len(jnp.where(~jnp.isnan(roots[m,:]))[0])>1) & (Is_create[m,:]!=0).any()):
        #parity更换符号的位置，并且该位置的下一个位置为nan（撞墙了）
        transit=jnp.where((parity[m,:]==-parity[m,n])&(jnp.isnan(roots[m_next,:])))[0][0]
        parity=parity.at[m,n].set(0)
        roots=roots.at[m,n].set(jnp.nan)
        m_map+=[m];n_map+=[transit]
        #将遍历过的位置设置为nan，将parity设置为0
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    #如果当前有一个不是nan，并且下一行也只有一个parity相同的,并且不是最后一行
    elif (len(jnp.where(parity[m_next,:]==parity[m,n])[0])==1):
        parity=parity.at[m,n].set(0)
        roots=roots.at[m,n].set(jnp.nan)
        m-=int(parity[m,n])
        transit=jnp.where((~jnp.isnan(roots[m,:])))[0][0]
        m_map+=[m];n_map+=[transit]
        m_map_res,n_map_res,temp_roots,temp_parity=search(m_map,n_map,roots,parity,fir_val,Is_create)
        return m_map_res,n_map_res,temp_roots,temp_parity
    else:
        parity=parity.at[m,n].set(0)
        roots=roots.at[m,n].set(jnp.nan)
        return m_map,n_map,roots,parity   
def search_first_postion(temp_roots,temp_parity):#搜索图像匹配开始的索引
    roots_now=temp_roots[0,:][~jnp.isnan(temp_roots[0,:])]
    change_sum=jnp.sum(temp_parity,axis=1)
    if (jnp.isin(1,change_sum).any()):#只处理parity 是 -1 1 -1的情况，否则转换为 -1 1 -1的情况
        temp_parity*=-1
    if jnp.shape(roots_now)[0]!=0:#如果第一行不是全部都为nan
        temp_cond=jnp.where((~jnp.isnan(temp_roots.real[0,:])) & (temp_parity[0,:]==-1))[0]#第一行存在的parity=sum(parity)的根
        initk=0
        if jnp.shape(temp_roots)[1]==2:#如果有两列
            try:
                initm=jnp.where(~jnp.isin(jnp.round(temp_roots[0,:],6),jnp.round(temp_roots[-1,:],6)))[0][0]#第一行不在最后一行的值
            except IndexError:
                initm=jnp.where(temp_parity[0,:]==-1)[0][0]
            if temp_parity[initk,initm]==1:
                temp_parity*=-1
        else:#第一行不是nan 并且不是两列
            Nan_idx=jnp.where(jnp.isnan(temp_roots[:,temp_cond]).any(axis=1))[0][0]
            initm=temp_cond[jnp.where(jnp.isnan(temp_roots[Nan_idx,temp_cond]))[0][0]]
    else:#如果有两列待链接,并且第一行全部为nan
        roots_last=temp_roots[-1,:][~jnp.isnan(temp_roots[-1,:])]#最后一行不是nan的地方
        if jnp.shape(roots_last)[0]==0:#如果最后一行都是nan
            initm=jnp.where((temp_parity==-1).any(axis=0))[0][0]
            initk=jnp.where(~jnp.isnan(temp_roots[:,initm]))[0][0]
        else:
            initk=-1
            initm=jnp.where(temp_parity[-1,:]==-1)[0][0]
            temp_parity*=-1
    return initk,initm,temp_parity