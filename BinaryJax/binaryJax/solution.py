import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from .util import Iterative_State,custom_insert,custom_delete
from .basic_function_jax import *
from .linear_sum_assignment_jax import find_nearest
from .polynomial_solver import get_roots
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
@jax.jit
def add_points(idx,add_zeta,add_theta,roots_State,s,m1,m2,add_number):
    sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create=roots_State
    add_coff = get_poly_coff(add_zeta,s,m2)
    add_roots,add_parity,add_ghost_roots,outloop,add_coff,add_zeta,add_theta,add_number=get_real_roots(add_coff,add_zeta,add_theta,s,m1,m2,add_number)#可能删掉不合适的根

    idx = jnp.where((add_number==0)[:,0],0,idx) ## if the add_number is deleted, then the idx is 0
    sample_n +=jnp.sum(add_number)

    insert_fun = lambda x,y,z: custom_insert(x,idx,y,add_number,z)
    theta,ghost_roots_dis = jax.tree_map(insert_fun, (theta,ghost_roots_dis), (add_theta,add_ghost_roots),(jnp.nan,)*2)

    buried_error=get_buried_error(ghost_roots_dis,sample_n)

    sort_flag = insert_fun(sort_flag,jnp.full(theta.shape,False),jnp.array([True]))

    unsorted_roots,unsorted_parity =jax.tree_map(insert_fun, (roots,parity),(add_roots,add_parity),(jnp.nan,)*2)

    roots,parity,sort_flag=get_sorted_roots(unsorted_roots,unsorted_parity,sort_flag)
    Is_create=find_create_points(roots,sample_n)
    return Iterative_State(sample_n,theta,roots,parity,ghost_roots_dis,sort_flag,Is_create),buried_error,outloop
@jax.jit
def get_buried_error(ghost_roots_dis,sample_n):
    n_ite=ghost_roots_dis.shape[0]
    error_buried=jnp.zeros((n_ite,1))

    idx_j = jnp.arange(1,n_ite)
    idx_i = jnp.roll(idx_j,shift=1)
    idx_k = jnp.roll(idx_j,shift=-1)

    idx_k = jnp.where(idx_k==sample_n,1,idx_k) # because the last point is the same as the first point 0=2pi
    idx_i = jnp.where(idx_i==n_ite-1,sample_n-1,idx_i)

    Ghost_i = ghost_roots_dis[idx_i]
    Ghost_j = ghost_roots_dis[idx_j]
    Ghost_k = ghost_roots_dis[idx_k]
    KInCaustic = (jnp.isnan(Ghost_k))
    IInCaustic = (jnp.isnan(Ghost_i))

    add_item = jnp.where((Ghost_i>2*Ghost_j)&(~KInCaustic),(Ghost_i-Ghost_j)**2,0)
    error_buried=error_buried.at[idx_k].add(add_item)

    add_item = jnp.where((2*Ghost_j<Ghost_k)&(~IInCaustic),(Ghost_k-Ghost_j)**2,0)
    error_buried=error_buried.at[idx_j].add(add_item)
    
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
def get_real_roots(coff,zeta_l,theta,s,m1,m2,add_number):
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
    outloop=0
    carry=lax.cond(((parity_sum!=-1)&mask).any(),parity_delete_cond,lambda x:x,(sample_n,theta,real_parity,real_roots,outloop,parity_sum,mask,add_number))
    sample_n,theta,real_parity,real_roots,outloop,parity_sum,_,add_number=carry
    ###计算得到最终的
    cond=(jnp.isnan(real_roots))&(jnp.arange(n_ite)<sample_n)[:,None]
    ghost_roots=jnp.where(cond,roots,jnp.inf)
    ghost_roots=jnp.sort(ghost_roots,axis=1)[:,0:2]
    ghost_roots=jnp.where(jnp.isinf(ghost_roots),jnp.nan,ghost_roots)
    ghost_roots_dis=jnp.abs(jnp.diff(ghost_roots,axis=1))
    return real_roots,real_parity,ghost_roots_dis,outloop,coff,zeta_l,theta,add_number
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
    real_parity = real_parity.at[-1].set(jnp.nan)
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
    parity_process_fun=lambda x: lax.cond((nan_num[i]==0)&(i<sample_n),parity_true1_fun,parity_false1_fun,x)
    real_parity= lax.cond(i<sample_n,parity_process_fun,lambda x:x[2],(temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2))
    # real_parity=lax.cond((nan_num[i]==0)&(i<sample_n),parity_true1_fun,parity_false1_fun,(temp,zeta_l,real_parity,i,cond,nan_num,s,m1,m2))
    return (zeta_l,real_roots,real_parity,nan_num,sample_n,cond,s,m1,m2),i
@jax.jit    
def find_create_points(roots, sample_n):
    """
    find the image are created or destroyed

    Parameters
    ----------
    roots : jnp.ndarray
        the roots of the polynomial
    sample_n : int
        the number of the sample points

    Returns
    -------
    jnp.ndarray
        the image are created or destroyed, if the image is created, the value is 1 and if the image is destroyed, the value is -1;
        if the create or destroy are on the same row, the connection is 10
        [ 1 , 1 , 1 , nan , nan ]
        [ 1 , 1 , 1 , 1 , 1 ] # create
         [ 1 , 1 , 1 , 1 , 1 ]  # destroy
        [ 1 , nan , nan , 1 , 1 ]
    """
    cond=jnp.isnan(roots)
    Is_create=jnp.zeros_like(roots,dtype=int)
    idx_x,idx_y=jnp.where(jnp.diff(cond,axis=0)&(~cond[0:-1].all(axis=1))[:,None]&(jnp.arange(roots.shape[0]-1)!=sample_n-1)[:,None],size=20,fill_value=-2)
    ## the roots at i is nan but the roots at i+1 is not nan/ the roots at i is not nan but the roots at i+1 is nan
    ## the index i can't be the last index
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
        create_value = lax.cond(~cond[x, y], create_value_true, create_value_false, None) ## if the root is not nan, then it is created
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
    sample_n,theta,real_parity,real_roots,outloop,parity_sum,mask,add_number=carry
    n_ite=theta.shape[0]
    cond = (parity_sum!=-1)& mask
    delidx=jnp.where(cond,size=n_ite,fill_value=10000)[0]

    sample_n-= cond.sum()
    ones_array = jnp.ones_like(delidx)
    ones_array = jnp.where(cond,0,ones_array)
    def fix_add_number_fun(carry,k):
        add_number,ones_array=carry
        old_number = add_number[k]
        add_number = add_number.at[k].set(
            jnp.where(jnp.arange(ones_array.shape[0])<add_number[k],ones_array,0).sum())
        ones_array = jnp.roll(ones_array,-old_number,axis=0)
        return (add_number,ones_array),k
    carry,_ = lax.scan(fix_add_number_fun,(add_number,ones_array),jnp.arange(add_number.shape[0]))
    add_number,_=carry

    delete_tree = [theta, real_parity, real_roots]
    theta,real_parity,real_roots=jax.tree_map(lambda x: custom_delete(x,delidx), delete_tree)
    
    outloop+=cond.sum()
    # if the parity is still wrong, then delete the point
    return (sample_n,theta,real_parity,real_roots,outloop,parity_sum,mask,add_number)