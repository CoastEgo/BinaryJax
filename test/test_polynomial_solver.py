from itertools import product

import jax
import jax.numpy as jnp
import pytest
from microlux.basic_function import get_poly_coff, to_lowmass
from microlux.polynomial_solver import Aberth_Ehrlich, AE_roots0
from test_util import get_caustic_permutation


rho_values = [1e-2, 1e-3, 1e-4]
q_values = [1e-1, 1e-2, 1e-3]
s_values = [0.6, 1.0, 1.4]


@pytest.mark.fast
@pytest.mark.parametrize("rho, q, s", product(rho_values, q_values, s_values))
def test_polynomial_caustic(rho, q, s):
    trajectory_c = get_caustic_permutation(rho, q, s, n_points=100)
    theta_sample = jnp.linspace(0, 2 * jnp.pi, 100)
    contours = (trajectory_c + rho * jnp.exp(1j * theta_sample)[:, None]).ravel()

    z_lowmass = to_lowmass(s, q, contours)

    coff = get_poly_coff(z_lowmass[:, None], s, q / (1 + q))

    get_AE_roots = lambda x: Aberth_Ehrlich(x, AE_roots0(x), MAX_ITER=50).sort()
    AE_roots = jax.jit(jax.vmap(get_AE_roots))(coff)

    get_numpy_roots = lambda x: jnp.roots(x, strip_zeros=False).sort()
    numpy_roots = jax.jit(jax.vmap(get_numpy_roots))(coff)

    error = jnp.abs(AE_roots - numpy_roots)

    print("max absolute error is", jnp.max(error))
    assert jnp.allclose(AE_roots, numpy_roots, atol=1e-10)


@pytest.mark.fast
@pytest.mark.parametrize("q, s", product(q_values, s_values))
def test_polynomial_uniform(q, s):
    x, y = jax.random.uniform(jax.random.PRNGKey(0), (2, 100000), minval=-2, maxval=2)

    trajectory_c = x + 1j * y
    z_lowmass = to_lowmass(s, q, trajectory_c)

    coff = get_poly_coff(z_lowmass[:, None], s, q / (1 + q))

    get_AE_roots = lambda x: Aberth_Ehrlich(x, AE_roots0(x), MAX_ITER=50).sort()
    AE_roots = jax.jit(jax.vmap(get_AE_roots))(coff)

    get_numpy_roots = lambda x: jnp.roots(x, strip_zeros=False).sort()
    numpy_roots = jax.jit(jax.vmap(get_numpy_roots))(coff)

    error = jnp.abs(AE_roots - numpy_roots)
    print("max absolute error is", jnp.max(error))

    assert jnp.allclose(AE_roots, numpy_roots, atol=1e-10)


if __name__ == "__main__":
    test_polynomial_caustic(1e-2, 0.2, 0.9)
    test_polynomial_uniform(0.2, 0.9)
