import jax
import jax.numpy as jnp
import numpy as np

from .basic_function import basic_partial, dot_product


def error_ordinary(deXProde2X, de_z, delta_theta, z, parity, de_deXPro_de2X):
    dAp_1 = 1 / 24 * (deXProde2X[0:-1] + deXProde2X[1:]) * delta_theta
    delta_theta_wave = jnp.abs(z[0:-1] - z[1:]) ** 2 / jnp.abs(
        dot_product(de_z[0:-1], de_z[1:])
    )

    dAp_v1 = (
        dAp_1 * delta_theta**2 * parity[0:-1]
    )  # old version of parabolic correction term
    dAp_v2 = (
        1
        / 12
        * (
            (z.real[1:] - z.real[0:-1]) * (de_z.imag[1:] - de_z.imag[0:-1])
            - (z.imag[1:] - z.imag[0:-1]) * (de_z.real[1:] - de_z.real[0:-1])
        )
        * delta_theta
        * parity[0:-1]
    )  # new version of parabolic correction term
    dAp = 0.5 * (dAp_v1 + dAp_v2)
    # dAp = dAp_v1

    # e1=jnp.abs(1/48*jnp.abs(jnp.abs(deXProde2X[0:-1]-jnp.abs(deXProde2X[1:])))*delta_theta**3) # old version
    e1 = jnp.abs(dAp_v1 - dAp_v2) * 0.5
    e2 = 3 / 2 * jnp.abs(dAp_1 * (delta_theta_wave - delta_theta**2))
    e3 = 1 / 10 * jnp.abs(dAp) * delta_theta**2

    de_dAp = (
        1
        / 24
        * (de_deXPro_de2X[0:-1] - de_deXPro_de2X[1:])
        * delta_theta**3
        * parity[0:-1]
    )  # the gradient of the parabolic correction term
    e4 = (
        1 / 10 * jnp.abs(de_dAp)
    )  ## e4 is the error item to estimate the gradient error of the parabolic correction term
    # jax.debug.print('{}',jnp.nansum(e4)))
    e_tot = e1 + e2 + e3 + e4
    return e_tot, dAp  # 抛物线近似的补偿项


def error_critial(pos_idx, neg_idx, i, create, parity, deXProde2X, z, de_z):
    # pos_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==-1*create),size=1)[0]
    # neg_idx=jnp.where(((Is_create[i]==create)|(Is_create[i]==10))&(parity[i]==1*create),size=1)[0]
    z_pos = z[i, pos_idx]
    z_neg = z[i, neg_idx]

    theta_wave = jnp.abs(z_pos - z_neg) / jnp.sqrt(
        jnp.abs(dot_product(de_z[i, pos_idx], de_z[i, neg_idx]))
    )

    dAcP_v1 = (
        parity[i, pos_idx]
        * 1
        / 24
        * (deXProde2X[i, pos_idx] - deXProde2X[i, neg_idx])
        * theta_wave**3
    )  # old version of parabolic correction term at the critical point
    dAcP_v2 = (
        -1
        / 12
        * (
            (z[i, neg_idx].real - z[i, pos_idx].real)
            * (de_z[i, pos_idx].imag + de_z[i, neg_idx].imag)
            - (z[i, neg_idx].imag - z[i, pos_idx].imag)
            * (de_z[i, pos_idx].real + de_z[i, neg_idx].real)
        )
        * theta_wave
        * parity[i, pos_idx]
    )  # new version of parabolic correction term at the critical point
    dAcP = 0.5 * (dAcP_v1 + dAcP_v2)

    # ce1=1/48*jnp.abs(deXProde2X[i,pos_idx]+deXProde2X[i,neg_idx])*theta_wave**3 # old version
    ce1 = 0.5 * jnp.abs(dAcP - dAcP_v2)  # new version of error term 1
    ce2 = (
        3
        / 2
        * jnp.abs(
            dot_product(z_pos - z_neg, de_z[i, pos_idx] - de_z[i, neg_idx])
            - create
            * 2
            * jnp.abs(z_pos - z_neg)
            * jnp.sqrt(jnp.abs(dot_product(de_z[i, pos_idx], de_z[i, neg_idx])))
        )
        * theta_wave
    )
    ce3 = 1 / 10 * jnp.abs(dAcP) * theta_wave**2
    ce_tot = ce1 + ce2 + ce3

    return (
        ce_tot,
        jnp.sum(dAcP),
        1
        / 2
        * (z[i, pos_idx].imag + z[i, neg_idx].imag)
        * (z[i, pos_idx].real - z[i, neg_idx].real),
    )  # critial 附近的抛物线近似'''


def error_sum(Roots_State, rho, q, s, mask=None):
    Is_create = Roots_State.Is_create
    z = Roots_State.roots
    parity = Roots_State.parity
    theta = Roots_State.theta
    if mask is None:
        mask = ~jnp.isnan(z)
    caustic_crossing = (Is_create[3, :] != 0).any()
    deXProde2X, de_z, de_deXPro_de2X = basic_partial(
        z, theta, rho, q, s, caustic_crossing
    )
    deXProde2X = jnp.where(mask, deXProde2X, 0.0)
    de_z = jnp.where(mask, de_z, 0.0)
    de_deXPro_de2X = jnp.where(mask, de_deXPro_de2X, 0.0)

    error_hist = jnp.zeros_like(theta)
    delta_theta = jnp.diff(theta, axis=0)

    mag = jnp.array([0.0])
    e_ord, parab = error_ordinary(
        deXProde2X, de_z, delta_theta, z, parity, de_deXPro_de2X
    )

    diff_mask = mask[1:] & mask[:-1]
    e_ord = jnp.where(diff_mask, e_ord, 0.0)
    parab = jnp.where(diff_mask, parab, 0.0)
    e_ord = jnp.sum(e_ord, axis=1)
    parab = jnp.sum(parab)

    error_hist = error_hist.at[1:].set(e_ord[:, None])

    de_z = jnp.where(
        mask, de_z, 10.0
    )  # avoid the nan value in the theta_wave calculation

    def no_create_true_fun(carry):
        ## if there is image create or destroy, we need to calculate the error

        mag, parab, error_hist = carry
        critial_row_idx, critial_pos_idx, critial_neg_idx, create_array = Is_create
        total_num = (create_array != 0).sum()
        result_row_idx = critial_row_idx
        critical_error, dApc, magc = jax.vmap(
            error_critial, in_axes=(0, 0, 0, 0, None, None, None, None)
        )(
            critial_pos_idx,
            critial_neg_idx,
            result_row_idx,
            create_array,
            parity,
            deXProde2X,
            z,
            de_z,
        )
        critical_error = jnp.where(
            jnp.arange(len(critical_error)) < total_num, critical_error, 0.0
        )
        magc = jnp.where(jnp.arange(len(magc)) < total_num, magc, 0.0)
        dApc = jnp.where(jnp.arange(len(dApc)) < total_num, dApc, 0.0)
        error_hist = error_hist.at[(result_row_idx - (create_array - 1) // 2), 0].add(
            critical_error
        )
        mag += jnp.sum(magc)
        parab += jnp.sum(dApc)

        return (mag, parab, error_hist)

    carry = jax.lax.cond(
        caustic_crossing,
        no_create_true_fun,
        lambda x: x,
        (0.0, 0.0, jnp.zeros_like(theta)),
    )
    mag_c, parab_c, error_hist_c = carry
    mag += mag_c
    parab += parab_c
    error_hist += error_hist_c

    return error_hist / (np.pi * rho**2), mag, parab
