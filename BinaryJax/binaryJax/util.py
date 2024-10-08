import jax
import jax.numpy as jnp
from jax import lax
from typing import NamedTuple,Union,Any,Optional
import warnings
MAX_CAUSTIC_INTERSECT_NUM = 15
class Iterative_State(NamedTuple):
    sample_num: int
    theta: jax.Array
    roots: jax.Array
    parity: jax.Array
    ghost_roots_distant: jax.Array
    sort_flag: Union[bool,jax.Array]
    Is_create: jax.Array = jnp.zeros([4,MAX_CAUSTIC_INTERSECT_NUM],dtype=int)

class Error_State(NamedTuple):
    mag: jax.Array
    mag_no_diff: int
    outloop: int
    error_hist: jax.Array
    epsilon: float
    epsilon_rel: float
    exceed_flag: bool = False

class Model_Param(NamedTuple):
    rho: float
    q: float
    s: float
    m1: float
    m2: float

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

def custom_insert(array,idx,add_array):
    final_array = jnp.insert(array,idx,add_array,axis=0)
    final_array = final_array[:array.shape[0]]
    return final_array

def delete_body(carry, k):
    array, ite2 ,delidx = carry
    mask = ite2 < delidx[k]
    array = jnp.where(mask[:,None], array, jnp.roll(array, -1,axis=0))
    delidx -= (~mask).any()
    return (array, ite2, delidx ), k

def custom_delete(array, delidx):
    fill_value = array[-1]
    ite = jnp.arange(array.shape[0])
    carry, _ = lax.scan(delete_body, (array, ite, delidx), jnp.arange(delidx.shape[0]))
    array, _, _ = carry
    array = jnp.where((ite < ite.size - (delidx<array.shape[0]).sum())[:,None], array, fill_value)
    return array

def stop_grad_wrapper(func):
    def wrapper(*args, **kwargs):
        args = jax.lax.stop_gradient(args)
        kwargs = jax.lax.stop_gradient(kwargs)
        return jax.lax.stop_gradient(func(*args, **kwargs))
    return wrapper

def warn_length_not_enough(required_length, Max_length):
    warnings.warn(
        "No enough space to insert new samplings, which may cause the error larger than the tolerance. Current length vs max length: {} vs {}. Consider incresing default_strategy parameters.".format(required_length, Max_length-2))