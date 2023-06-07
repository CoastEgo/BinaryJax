import numpy as np
import jax
import scipy as sp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
from jax import jvp, grad
from jax import numpy as jnp
from jax import custom_jvp
from jax import jacfwd
from jax import lax
import time
def loop_body(carry):
    coff,der1,der2,xk,n,epsilon=carry
    G = jnp.polyval(der1,xk) / jnp.polyval(coff,xk)
    H = G ** 2 - jnp.polyval(der2,xk) / jnp.polyval(coff,xk)
    root = jnp.sqrt((n - 1) * (n * H - G ** 2))
    temp=jnp.stack([G + root, G - root])
    d = temp[jnp.argmax(jnp.abs(temp))]
    a = n / d
    xk -= a
    return (coff,der1,der2,xk,n,epsilon)
def cond_fun(carry):
    coff,der1,der2,xk,n,epsilon=carry
    return jnp.abs(jnp.polyval(coff,xk)) > epsilon
@jax.jit
def laguerre_method(coff,x0,n, epsilon= 1e-14):
    der1=jnp.polyder(coff,1)
    der2=jnp.polyder(coff,2)
    xk = x0
    carry=lax.while_loop(cond_fun,loop_body,(coff,der1,der2,xk,n,epsilon))
    coff,der1,der2,xk,n,epsilon=carry
    return xk
@jax.jit
def zroots(coff):
    roots=jnp.empty(coff.shape[0]-1,dtype=jnp.complex128)
    def body_fun(carry,k):
        coff,roots=carry
        roots=roots.at[k].set(laguerre_method(coff,0,5-k))#order of polynomial
        coff,_=jnp.polydiv(coff,jnp.array([1,-1*roots[k]]))
        coff=jnp.resize(coff,(6,))#number of coffs
        coff=jnp.where(jnp.arange(6)<1,0.,jnp.roll(coff,1))
        return(coff,roots),k
    carry,_=lax.scan(body_fun,(coff,roots),jnp.arange(coff.shape[0]-1))
    '''for i in range(coff.shape[0]-1):
        carry,_=body_fun((coff,roots),i)
        coff,roots=carry'''
    coff,roots=carry
    return roots
# closed-form roots for quadratic, cubic, and quartic polynomials
# multi_quadratic and multi_quartic adapted from https://github.com/NKrvavica/fqs
# fast_cubic rewritten for complex polynomials
# https://arxiv.org/abs/2207.12412
# author: Keming Zhang
@jax.jit
def multi_quadratic(a0, b0, c0):
    ''' Analytical solver for multiple quadratic equations
    (2nd order polynomial), based on `numpy` functions.
    Parameters
    ----------
    a0, b0, c0: array_like
        Input data are coefficients of the Quadratic polynomial::
            a0*x^2 + b0*x + c0 = 0
    Returns
    -------
    r1, r2: ndarray
        Output data is an array of two roots of given polynomials.
    '''
    ''' Reduce the quadratic equation to to form:
        x^2 + ax + b = 0'''
    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = jnp.sqrt(delta + 0j)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return jnp.concatenate((r1, r2))
@jax.jit
def fast_cubic(a, b, c, d):
    d0 = b**2-3*a*c
    d1 = 2*b**3-9*a*b*c+27*a**2*d
    d2 = jnp.sqrt(d1**2-4*d0**3+0j)
    
    d3 = d1-d2
    mask = d2 == d1
    d3=jnp.where(mask,d3+d2,d3)[0]
    #d3[mask] += 2*d2[mask]
    
    C = (d3/2)**(1/3)
    d4 = d0/C
    #d4[(C == 0)*(d0 == 0)] = 0
    d4=jnp.where((C==0)&(d0==0),0,d4)[0]
    pcru = (-1-(-3)**0.5)/2
    #roots
    x0 = -1/3/a*(b+C+d4)
    x1 = -1/3/a*(b+C*pcru+d4/pcru)
    x2 = -1/3/a*(b+C*pcru**2+d4/pcru**2)
    return jnp.concatenate((x0, x1, x2))
@jax.jit
def multi_quartic(a0, b0, c0, d0, e0):
    ''' Analytical closed-form solver for multiple quartic equations
    (4th order polynomial), based on `numpy` functions. Calls
    `multi_cubic` and `multi_quadratic`.
    Parameters
    ----------
    a0, b0, c0, d0, e0: array_like
        Input data are coefficients of the Quartic polynomial::
            a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0
    Returns
    -------
    r1, r2, r3, r4: ndarray
        Output data is an array of four roots of given polynomials.
    '''

    ''' Reduce the quartic equation to to form:
        x^4 ax^3 + bx^2 + cx + d = 0'''
    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = fast_cubic(1, p, r, p*r - 0.5*q*q)[0]

    # Additional variables
    s = jnp.sqrt(2*p + 2*z0 + 0j)
    t = jnp.zeros_like(s)
    mask = (jnp.abs(s) < 1e-8)
    t=jnp.where(mask,z0*z0+r,-q/s)[0]
    #t[mask] = z0[mask]*z0[mask] + r[mask]
    #t[~mask] = -q[~mask] / s[~mask]

    # Compute roots by quadratic equations
    r01 = multi_quadratic(1, s, z0 + t) - a0
    r23 = multi_quadratic(1, -s, z0 - t) - a0

    return jnp.concatenate((r01, r23))
@jax.jit
def halfanalytical(coff):
    roots=jnp.empty(coff.shape[0]-1,dtype=jnp.complex128)
    roots=roots.at[0].set(laguerre_method(coff,0,5))#order of polynomial
    coff,_=jnp.polydiv(coff,jnp.array([1,-1*roots[0]]))
    roots=roots.at[1:].set(multi_quartic(coff[0:1],coff[1:2],coff[2:3],coff[3:4],coff[4:]))
    return roots
'''coff=np.random.rand(6)+1j*np.random.rand(6)
a=np.poly1d(coff)
st=time.perf_counter()
for i in range(1):
    npr=a.roots
end=time.perf_counter()
print('numpy roots time=',end-st)
zr=zroots(coff)
st=time.perf_counter()
for i in range(1):
    zr=zroots(coff)
end=time.perf_counter()
print('laguerre time jitted=',end-st)
print(jnp.polyval(coff,npr))
print(jnp.polyval(coff,zr))'''
'''coff=np.random.rand(6)+1j*np.random.rand(6)*5
s=time.perf_counter()
for i in range(10000):
    npr=np.roots(coff)
e=time.perf_counter()
print('numpy time',e-s)
print('numpy error',np.abs(np.polyval(coff,npr)))
zr=zroots(coff)
s=time.perf_counter()
for i in range(10000):
    zr=zroots(coff)
e=time.perf_counter()
zrtime=e-s
print('laguerre time',zrtime)
print('laguerre error',np.abs(np.polyval(coff,zr)))
halfz=halfanalytical(coff)
s=time.perf_counter()
for i in range(10000):
    halfz=halfanalytical(coff)
e=time.perf_counter()
print('half laguerre time',e-s)
print('half laguerre error',np.abs(np.polyval(coff,halfz)))
print((e-s)/zrtime)'''