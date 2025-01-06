# microlux

This code is a Jax-based package that is used to calculate the binary lensing light curve with the finite source effect using the contour integration method.  We inherit the novel features in <a href='https://github.com/valboz/VBBinaryLensing'>VBBinaryLensing</a> including parabolic correction and optimal sampling to maximize the performance. This is built on the <a href='https://github.com/google/jax'>JAX</a> library which provides a NumPy-like interface with GPU and automatic differentiation support for high-performance machine learning research. Through automatic differentiation and our package, we get access to the accurate gradient for exploring more advanced algorithms. 
## Installation

You can install this package from the source by cloning the repository and running it after installing Jax. For the installation of Jax, please refer to the <a href='https://github.com/google/jax#installation'>Jax installation guide. </a>

``` bash 

git clone https://github.com/CoastEgo/microlux.git

cd microlux

pip install -e .

```
## Usage

The Jax version can calculate the gradient of the light curve using <a href='https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html'>automatic differentiation</a>.
There are examples in the ```test``` folder which show the light curve and gradient calculation.

## Applications

We can combine the Jax code with <a href='https://github.com/pyro-ppl/numpyro'>Numpyro</a> for Hamiltonian Monte Carlo (HMC) sampling. There is a Jupyter notebook showing the application of HMC in modeling real microlensing events KMT-2019-BLG-0371 in the `example` folder.
## Features

- [x] Optimal sampling and contour integration with error control to calculate the binary lens microlensing light curve with finite source effect.
- [x] Robust and accurate calculation: Widely test over the broad parameter space compared with VBBinaryLensing
- [x] Fast speed: Fully compatible with JIT compilation to speed up calculation. 
- [x] Accurate gradient: Automatic differentiation with novel error estimator to ensure the convergence of gradient. 
- [x] Aberth–Ehrlich method to find the roots of the polynomial and liner sum assignment algorithm to match the images
- [x] Application on real events modeling using NUTS in Numpyro

  

## Future work

- [ ] High-order effects: parallax, orbital motion etc.
- [ ] Machine learning Applications.

  

## Reference

*under development* and Paper is coming soon.