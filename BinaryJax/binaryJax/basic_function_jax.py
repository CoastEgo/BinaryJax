import numpy as np
import jax.numpy as jnp
from .polynomial_solver import Aberth_Ehrlich,AE_roots0
import jax
from functools import partial
from jax import lax
from jax import custom_jvp
from .linear_sum_assignment_jax import solve
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
@jax.jit
def Quadrupole_test(rho,s,q,zeta,z,cond,tol=1e-2):
    m1=1/(1+q)
    m2=q/(1+q)
    cQ=6;cG=2;cP=2
    fz0 = lambda z: -m1/(z-s)-m2/z
    fz1 = lambda z: m1/(z-s)**2+m2/z**2
    fz2 = lambda z: -2*m1/(z-s)**3-2*m2/z**3
    fz3 = lambda z: 6*m1/(z-s)**4+6*m2/z**4
    J = lambda z: 1-fz1(z)*jnp.conj(fz1(z))
    ####Quadrupole test
    miu_Q=jnp.abs(-2*jnp.real(3*jnp.conj(fz1(z))**3*fz2(z)**2-(3-3*J(z)+J(z)**2/2)*jnp.abs(fz2(z))**2+J(z)*jnp.conj(fz1(z))**2*fz3(z))/(J(z)**5))
    miu_C=jnp.abs(6*jnp.imag(3*jnp.conj(fz1(z))**3*fz2(z)**2)/(J(z)**5))
    mag=jnp.sum(jnp.where(cond, jnp.abs(1/J(z)), 0), axis=1)
    cond1=jnp.sum(jnp.where(cond, miu_Q+miu_C, 0), axis=1)*cQ*(rho**2+1e-4*tol)<tol
    ####ghost image test
    zwave=jnp.conj(zeta)-fz0(z)
    J_wave=1-fz1(z)*fz1(zwave)
    miu_G=1/2*jnp.abs(J(z)*J_wave**2/(J_wave*fz2(jnp.conj(z))*fz1(z)-jnp.conj(J_wave)*fz2(z)*fz1(jnp.conj(z))*fz1(zwave)))
    miu_G = jnp.where(cond, 100, miu_G)
    cond2=((cG*(rho+1e-3)<miu_G).all(axis=1))#any更加宽松，因为ghost roots应该是同时消失的，理论上是没问题的
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
    zeta_l=trajectory_centroid_l+rho*jnp.exp(1j*theta)
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
@jax.jit
def dot_product(a,b):
    return jnp.real(a)*jnp.real(b)+jnp.imag(a)*jnp.imag(b)

@jax.jit
def basic_partial(z,theta,rho,q,s):
    z_c=jnp.conj(z)
    parZetaConZ=1/(1+q)*(1/(z_c-s)**2+q/z_c**2)
    par2ConZetaZ=-2/(1+q)*(1/(z-s)**3+q/(z)**3)
    de_zeta=1j*rho*jnp.exp(1j*theta)
    detJ=1-jnp.abs(parZetaConZ)**2
    de_z=(de_zeta-parZetaConZ*jnp.conj(de_zeta))/detJ
    deXProde2X=(rho**2+jnp.imag(de_z**2*de_zeta*par2ConZetaZ))/detJ

    # calculate the derivative of x'^x'' with respect to \theta x'^x'''
    de2_zeta = -rho*jnp.exp(1j*theta)
    de2_zetaConj = -rho*jnp.exp(-1j*theta)
    par3ConZetaZ=6/(1+q)*(1/(z-s)**4+q/(z)**4)
    de2_z = (de2_zeta-jnp.conj(par2ConZetaZ)*jnp.conj(de_z)**2-parZetaConZ*(
        de2_zetaConj-par2ConZetaZ*de_z**2))/detJ
    
    # deXProde2X_test = 1/(2*1j)*(de2_z*jnp.conj(de_z)-de_z*jnp.conj(de2_z))
    # jax.debug.print('deXProde2X_test error is {}',jnp.nansum(jnp.abs(deXProde2X_test-deXProde2X)))
    de_deXPro_de2X=1/detJ**2*jnp.imag(
        detJ*(de2_zeta*par2ConZetaZ*de_z**2+de_zeta*par3ConZetaZ*de_z**3+de_zeta*par2ConZetaZ*2*de_z*de2_z)
        +(jnp.conj(par2ConZetaZ)*jnp.conj(de_z)*jnp.conj(parZetaConZ)+parZetaConZ*par2ConZetaZ*de_z
        )*de_zeta*par2ConZetaZ*de_z**2)
    # deXProde2X = jax.lax.stop_gradient(deXProde2X)
    return deXProde2X,de_z,de_deXPro_de2X