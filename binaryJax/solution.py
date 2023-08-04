import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from .basic_function_jax import *
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
@jax.jit
def add_points(idx,add_zeta,add_coff,add_theta,roots,parity,theta,ghost_roots_dis,sort_flag,s,m1,m2,sample_n,add_number):
    add_roots,add_parity,add_ghost_roots,outloop,add_coff,add_zeta,add_theta=get_real_roots(add_coff,add_zeta,add_theta,s,m1,m2)#可能删掉不合适的根
    theta=custom_insert(theta,idx,add_theta,add_number)
    ghost_roots_dis=custom_insert(ghost_roots_dis,idx,add_ghost_roots,add_number)
    buried_error=get_buried_error(ghost_roots_dis)
    sort_flag=custom_insert(sort_flag,idx,jnp.array([False])[:,None],add_number)
    roots,parity,sort_flag=get_sorted_roots(custom_insert(roots,idx,add_roots,add_number),custom_insert(parity,idx,add_parity,add_number),sort_flag)
    Is_create=find_create_points(roots,sample_n)
    return theta,ghost_roots_dis,buried_error,sort_flag,roots,parity,Is_create,outloop
@jax.jit
def get_buried_error(ghost_roots_dis):
    n_ite=ghost_roots_dis.shape[0]
    error_buried=jnp.zeros((n_ite,1))
    idx1=jnp.where(ghost_roots_dis[0:-2]>2*ghost_roots_dis[1:-1],size=20,fill_value=-3)[0]+1#i-(i+1)>(i+1)对应的i+1
    idx1=jnp.where(~jnp.isnan(ghost_roots_dis[idx1+1]),idx1,-2)#i+2对应的不是nan，说明存在buried image
    error_buried=error_buried.at[idx1+1].add((ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2)#在i+2处加入误差项，因为应该在i+1，i+2处加点
    error_buried=error_buried.at[idx1].add((ghost_roots_dis[idx1]-ghost_roots_dis[idx1-1])**2)#在i+1处加入误差项，防止buried误差不收敛
    idx1=jnp.where(2*ghost_roots_dis[1:-1]<ghost_roots_dis[2:],size=20,fill_value=-3)[0]+1#i<i+1-i对应的i
    idx1=jnp.where(~jnp.isnan(ghost_roots_dis[idx1-1]),idx1,-2)#i-1处不是nan
    error_buried=error_buried.at[idx1].add((ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2)#在i处加入误差，也就是在i,i+1处加点
    error_buried=error_buried.at[idx1+1].add((ghost_roots_dis[idx1+1]-ghost_roots_dis[idx1])**2)#在i+1处加入误差项，防止buried误差不收敛
    error_buried.at[-2:].set(0.)
    error_buried=jnp.where(jnp.isnan(error_buried),0,error_buried)
    return error_buried
@jax.jit
def get_sorted_roots(roots,parity,sort_flag):
    roots.at[-2:].set(0.)
    parity.at[-2:].set(0.)
    flase_i=jnp.where(~sort_flag,size=roots.shape[0],fill_value=-1)[0]
    carry,_=lax.scan(sort_body1,(roots, parity),flase_i)
    resort_i=jnp.where((~sort_flag[0:-1])&(sort_flag[1:]),size=roots.shape[0]-1,fill_value=-2)[0]
    carry,_=lax.scan(sort_body2,carry,resort_i)
    roots,parity=carry
    sort_flag=sort_flag.at[:].set(True)
    roots.at[-2:].set(jnp.nan)
    parity.at[-2:].set(jnp.nan)
    return roots,parity,sort_flag
@jax.jit
def get_real_roots(coff,zeta_l,theta,s,m1,m2):
    n_ite=zeta_l.shape[0]
    sample_n=(~jnp.isnan(zeta_l)).any(axis=1).sum()
    mask=(jnp.arange(n_ite)<sample_n)
    roots=get_roots(n_ite,jnp.where(mask[:,None],coff,0.))#求有效的根
    roots=jnp.where(mask[:,None],roots,jnp.nan)
    parity=get_parity(roots,s,m1,m2)
    error=verify(zeta_l,roots,s,m1,m2)
    cond=error>1e-6
    nan_num=cond.sum(axis=1)
    ####计算verify,如果parity出现错误或者nan个数错误，则重新规定error最大的为nan
    #idx_verify_wrong=jnp.where(((nan_num!=0)&(nan_num!=2)),jnp.arange(n_ite),jnp.nan)#verify出现错误的根的索引
    idx_verify_wrong=jnp.where((nan_num!=0)&(nan_num!=2),size=n_ite,fill_value=-1)[0]#verify出现错误的根的索引,填充-1以保持数组形状，最后对-1单独处理即可
    cond=lax.cond((idx_verify_wrong!=-1).any(),update_cond,lambda x:x[-1],(idx_verify_wrong,error,cond))
    ####根的处理
    nan_num=cond.sum(axis=1)##对于没有采样到的位置也是0
    real_roots=jnp.where(cond,jnp.nan+jnp.nan*1j,roots)
    real_parity=jnp.where(cond,jnp.nan,parity)
    parity_sum=jnp.nansum(real_parity,axis=1)
    idx_parity_wrong=jnp.where((parity_sum!=-1)&mask,size=n_ite,fill_value=-1)[0]#parity计算出现错误的根的索引
    real_parity=lax.cond((idx_parity_wrong!=-1).any(),update_parity,lambda x:x[-1],(zeta_l,real_roots,nan_num,sample_n,idx_parity_wrong,cond,s,m1,m2,real_parity))
    parity_sum=jnp.nansum(real_parity,axis=1)
    outloop=False
    carry=lax.cond(((parity_sum!=-1)&mask).any(),parity_delete_cond,lambda x:x,(sample_n,theta,real_parity,real_roots,outloop,parity_sum))
    sample_n,theta,real_parity,real_roots,outloop,parity_sum=carry
    ###计算得到最终的
    cond=(jnp.isnan(real_roots))&(jnp.arange(n_ite)<sample_n)[:,None]
    ghost_roots=jnp.where(cond,roots,jnp.inf)
    ghost_roots=jnp.sort(ghost_roots,axis=1)[:,0:2]
    ghost_roots=jnp.where(jnp.isinf(ghost_roots),jnp.nan,ghost_roots)
    ghost_roots_dis=jnp.abs(jnp.diff(ghost_roots,axis=1))
    return real_roots,real_parity,ghost_roots_dis,outloop,coff,zeta_l,theta
@jax.jit
def update_cond(carry):
    idx_verify_wrong,error,cond=carry
    sorted=jnp.argsort(error[idx_verify_wrong],axis=1)[:,-2:]
    cond=cond.at[idx_verify_wrong].set(False)
    cond=cond.at[idx_verify_wrong,sorted[:,0]].set(True)
    cond=cond.at[idx_verify_wrong,sorted[:,1]].set(True)
    cond=cond.at[-1].set(False)
    return cond
    ##对于parity计算错误的点，分为fifth principal left center right，其中left center right 的parity为-1，1，-1
def update_parity(carry):
    zeta_l,real_roots,nan_num,sample_n,idx_parity_wrong,cond,s,m1,m2,real_parity=carry
    carry,_=lax.scan(loop_parity_body,(zeta_l,real_roots,real_parity,nan_num,sample_n,cond,s,m1,m2),idx_parity_wrong)
    zeta_l,real_roots,real_parity,nan_num,sample_n,cond,s,m1,m2=carry
    real_parity.at[-1].set(jnp.nan)
    return real_parity
@jax.jit
def parity_true1_fun(carry):##对于5个根怎么判断其parity更加合理
    temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2=carry
    prin_idx=jnp.where(jnp.sign(temp.imag)==jnp.sign(zeta_l.imag[i]),size=1,fill_value=0)[0]#主图像的索引
    prin_root=temp[prin_idx][jnp.newaxis][0]
    prin_root=jnp.concatenate([prin_root,temp[jnp.argmax(get_parity_error(temp,s,m1,m2))][jnp.newaxis]])
    other=jnp.setdiff1d(temp,prin_root,size=3)
    x_sort=jnp.argsort(other.real)
    real_parity=real_parity.at[i,jnp.where((temp==other[x_sort[0]])|(temp==other[x_sort[-1]]),size=2)[0]].set(-1)
    real_parity=real_parity.at[i,jnp.where((temp==other[x_sort[1]]),size=1)[0]].set(1)
    return real_parity
@jax.jit
def parity_false1_fun(carry):##对于3个根怎么判断其parity更加合理
    temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2=carry
    real_parity=lax.cond((nan_num[i]!=0)&((jnp.abs(zeta_l.imag[i])>1e-5)[0]),parity_true2_fun,lambda x:x[-2],(temp,zeta_l,cond,real_parity,i))
    return real_parity
@jax.jit
def parity_true2_fun(carry):##通过主图像判断，与zeta位于y轴同一侧的为1
    temp,zeta_l,cond,real_parity,i=carry
    real_parity=real_parity.at[i,jnp.where(~cond[i],size=3)].set(-1)
    real_parity=real_parity.at[i,jnp.where(jnp.sign(temp.imag)==jnp.sign(zeta_l.imag[i]),size=1)[0]].set(1)
    return real_parity
@jax.jit
def loop_parity_body(carry,i):##循环体
    zeta_l,real_roots,real_parity,nan_num,sample_n,cond,s,m1,m2=carry
    temp=real_roots[i]
    real_parity=lax.cond((nan_num[i]==0)&(i<sample_n),parity_true1_fun,parity_false1_fun,(temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2))
    return (zeta_l,real_roots,real_parity,nan_num,sample_n,cond,s,m1,m2),i
@jax.jit    
def find_create_points(roots, sample_n):
    cond=jnp.isnan(roots)
    Is_create=jnp.zeros_like(roots,dtype=int)
    idx_x,idx_y=jnp.where(jnp.diff(cond,axis=0)&(~cond[0:-1].all(axis=1))[:,None]&(jnp.arange(roots.shape[0]-1)!=sample_n-1)[:,None],size=20,fill_value=-2)
    idx_x+=1
    @jax.jit
    def update_is_create(carry, inputs):
        x, y = inputs
        cond, Is_create = carry
        def create_value_true(_):
            return 1
        def create_value_false(_):
            return lax.cond(cond[x, y] & (Is_create[x - 1, y] != 1),
                            lambda _: -1,
                            lambda _: 10,
                            None)
        create_value = lax.cond(~cond[x, y], create_value_true, create_value_false, None)
        Is_create = Is_create.at[lax.cond(~cond[x, y], lambda _: x, lambda _: x - 1, None), y].set(create_value)
        return (cond, Is_create), (x, y)
    initial_carry = (cond, Is_create)
    final_carry, _ = lax.scan(update_is_create, initial_carry, (idx_x, idx_y))
    Is_create = final_carry[1]
    Is_create=Is_create.at[-3:].set(0)
    return Is_create
@jax.jit
def sort_body1(values,k):
    roots, parity = values
    @jax.jit
    def False_fun_sort1(carry):
        roots,parity,i=carry
        sort_indices = find_nearest(roots[k - 1, :], parity[k - 1, :], roots[k, :], parity[k, :])
        roots = roots.at[k, :].set(roots[k, sort_indices])
        parity = parity.at[k, :].set(parity[k, sort_indices])
        return roots,parity
    carry=lax.cond(jnp.isnan(roots[k]).all(),lambda x:x[0:-1],False_fun_sort1,(roots,parity,k))
    roots,parity=carry
    return (roots,parity),k
@jax.jit
def sort_body2(carry,i):
    @jax.jit
    def False_fun(carry):
        roots,parity,i=carry
        sort_indices=find_nearest(roots[i],parity[i],roots[i+1],parity[i+1])
        cond = jnp.tile(jnp.arange(roots.shape[0])[:, None], (1, roots.shape[1])) < i+1
        roots=jnp.where(cond,roots,roots[:,sort_indices])
        parity=jnp.where(cond,parity,parity[:,sort_indices])
        return roots,parity
    roots,parity=carry
    carry=lax.cond(jnp.isnan(roots[i]).all(),lambda x:x[0:-1],False_fun,(roots,parity,i))
    roots,parity=carry
    return (roots,parity),i
@jax.jit
def parity_delete_cond(carry):
    sample_n,theta,real_parity,real_roots,outloop,parity_sum=carry
    n_ite=theta.shape[0]
    delidx=jnp.where(parity_sum!=-1,size=n_ite,fill_value=n_ite+1)[0]
    sample_n-=jnp.size(delidx)
    theta=custom_delete(theta,delidx)
    real_parity=custom_delete(real_parity,delidx)
    real_roots=custom_delete(real_roots,delidx)
    outloop=True
    return (sample_n,theta,real_parity,real_roots,outloop,parity_sum)