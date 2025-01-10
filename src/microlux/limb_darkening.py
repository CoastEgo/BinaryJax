import abc

from jax import numpy as jnp


class AbstractLimbDarkening:
    """

    Abstract class for limb darkening model. The limb darkening model should have two methods:
    - profile: the limb darkening profile at a given radius. This should be normalized with the average intensity = 1
    - cumulative_profile: the cumulative limb darkening profile at a given radius integrated from the center to the r
    """

    @abc.abstractmethod
    def profile(self, r: jnp.ndarray) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def cumulative_profile(self, r: jnp.ndarray) -> jnp.ndarray:
        pass


class LinearLimbDarkening(AbstractLimbDarkening):
    """
    linear limb darkening model with the integration normalized to 1
    """

    a: float

    def __init__(self, a: float):
        self.a = a

    def profile(self, r: jnp.ndarray) -> jnp.ndarray:
        return 1 / (1 - self.a / 3) * (1 - self.a * (1 - jnp.sqrt(1 - r**2)))

    def cumulative_profile(self, r: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.sqrt(1 - r**2)

        res = (self.a * (mu**2 - 2 / 3 * mu**3) - mu**2) / (1 - self.a / 3) + 1
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r = jnp.linspace(0, 1, 100)
    ld = LinearLimbDarkening(1)
    plt.plot(r, ld.profile(r) / ld.cumulative_profile(1), label="limb fun")
    plt.plot(r, ld.cumulative_profile(r), label="cumulative limb fun")
    plt.legend()
    plt.show()
