import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import VBBinaryLensing
from BinaryJax import contour_integral, to_lowmass
from jax import random
from MulensModel import caustics


def test_extend_sorce(rho, q, s, retol=1e-3):
    """
        Test around the caustic, apadpted from https://github.com/fbartolic/caustics/blob/main/tests/test_extended_source.py

    x    Parameters:
        -----------
        rho : float
            The radius of the source.
        q : float
            The mass ratio of the binary lens.
        s : float
            The separation of the binary lens components.
        retol : float, optional
            The relative tolerance for the VBBinaryLensing calculations (default is 1e-3).
    """

    caustic_1 = caustics.Caustics(q, s)
    x, y = caustic_1.get_caustics()
    z_centeral = jnp.array(jnp.array(x) + 1j * jnp.array(y))
    ## random change the position of the source
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = jax.random.uniform(subkey1, z_centeral.shape, minval=-np.pi, maxval=np.pi)
    r = random.uniform(subkey2, z_centeral.shape, minval=0.0, maxval=2 * rho)
    z_centeral = z_centeral + r * np.exp(1j * phi)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax.scatter(z_centeral.real, z_centeral.imag, s=0.5, color="orange", label="source")
    circle = plt.Circle(
        (z_centeral.real[0], z_centeral.imag[0]), rho, fill=False, color="green"
    )  # 创建一个圆，中心在 (z.real, z.imag)，半径为 rho
    ax.add_patch(circle)  # 将圆添加到坐标轴上
    ax.scatter(x, y, s=0.5, color="deepskyblue", label="caustic")
    ax.set_aspect("equal")

    ### change the coordinate system
    z_lowmass = to_lowmass(s, q, z_centeral)
    trajectory_n = z_centeral.shape[0]

    ### change the coordinate system
    VBBL = VBBinaryLensing.VBBinaryLensing()
    VBBL.RelTol = retol
    VBBL_mag = []
    start = time.time()
    for i in range(trajectory_n):
        VBBL_mag.append(
            VBBL.BinaryMag2(s, q, z_centeral.real[i], z_centeral.imag[i], rho)
        )
    print(f"average VBBL time={(time.time()-start)/trajectory_n}")
    VBBL_mag = np.array(VBBL_mag)

    Jax_mag = []
    ## compile time
    mag = contour_integral(
        z_lowmass[0], retol, retol, rho, s, q, default_strategy=(60, 240, 480)
    )[0]
    ## real time
    start = time.time()
    for i in range(trajectory_n):
        mag = contour_integral(
            z_lowmass[i], retol, retol, rho, s, q, default_strategy=(60, 240, 480)
        )[0]
        Jax_mag.append(mag)
    jax.block_until_ready(Jax_mag)
    print(f"average Jax time={(time.time()-start)/trajectory_n}")
    Jaxmag = np.array(Jax_mag)
    print(f"rho={rho},max error={np.max(np.abs(Jaxmag-VBBL_mag)/VBBL_mag)}")
    ax2.plot(np.abs(Jaxmag - VBBL_mag) / VBBL_mag)
    # ax2.plot(Jax_mag,color='r',label='binaryJax')
    # ax2.plot(VBBL_mag,color='b',label='VBBL')
    ## set log scale
    ax2.set_yscale("log")
    ax2.set_ylabel("relative error")
    plt.savefig(f"picture/extendtest_{rho}.png")


if __name__ == "__main__":
    rho = [1e-2, 1e-3]
    for i in range(len(rho)):
        test_extend_sorce(rho[i], 1e-3, 1)
