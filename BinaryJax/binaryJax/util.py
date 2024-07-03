import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple,Union,Any,Optional

class Iterative_State(NamedTuple):
    sample_num: int
    theta: jax.Array
    roots: jax.Array
    parity: jax.Array
    ghost_roots_distant: jax.Array
    sort_flag: Union[bool,jax.Array]
    Is_create: jax.Array

class Error_State(NamedTuple):
    mag: jax.Array
    mag_no_diff: int
    outloop: int
    error_hist: jax.Array
    epsilon: float
    epsilon_rel: float

class Model_Param(NamedTuple):
    rho: float
    q: float
    s: float
    m1: float
    m2: float
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
def custom_insert(array,idx,add_array,add_number,pad_item):
    add_array = jnp.pad(add_array, ((0, array.shape[0]-add_array.shape[0]), (0, 0)), mode='constant', constant_values=pad_item)
    carry,_=lax.scan(insert_body,(array,add_array,idx,add_number),jnp.arange(idx.shape[0]))
    array,add_array,idx,add_number=carry
    return array
# @jax.jit
# def theta_encode(carry,k):
#     (theta,idx,add_number,add_theta_encode)=carry
#     add_total_num = 30
#     theta_diff = (theta[idx[k]] - theta[idx[k]-1]) / (add_number[k]+1)
#     add_theta=jnp.arange(1,add_total_num+1)[:,None]*theta_diff+theta[idx[k]-1]
#     add_theta=jnp.where((jnp.arange(add_total_num)<add_number[k])[:,None],add_theta,jnp.nan)
#     carry2,_=insert_body((add_theta_encode,add_theta,jnp.where(jnp.isnan(add_theta_encode),size=1)[0],add_number[k][None]),0)
#     add_theta_encode=carry2[0]
#     return (theta,idx,add_number,add_theta_encode),k
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

def stop_grad_wrapper(func):
    def wrapper(*args, **kwargs):
        args = jax.lax.stop_gradient(args)
        kwargs = jax.lax.stop_gradient(kwargs)
        return jax.lax.stop_gradient(func(*args, **kwargs))
    return wrapper