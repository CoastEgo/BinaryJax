# BinaryJax
This code is a auto-diff version of the <a href='https://github.com/valboz/VBBinaryLensing'> VBozza's BinaryLensing</a>.It is built on the <a href='https://github.com/google/jax'> JAX</a> library which provides a NumPy-like interface with GPU support and automatic differentiation for high-performance machine learning research.

## Installation
You can install it from source by cloning the repository and running after installing Jax.For how to install Jax, please refer to <a href='https://github.com/google/jax#installation'> Jax installation guide. </a>   
```git clone https://github.com/CoastEgo/microlensing.git```  
```cd microlensing```  
```pip install -e .```  

## Usage
There are examples in the ```examples``` folder. You can run them directly can change the input parameters by yourself.
### Numpy version
There are two versions of the code: numpy version and jax version. The difference is that numpy version can return the image curve while jax version can not. For the reason that Jax is hard to debug and its JIT compilation overhead, you can test it with VBBL to avoid VBBL's <a href='https://github.com/valboz/VBBinaryLensing/blob/master/docs/AdvancedControl.md'> potential failures </a>. It is better to do this when you use VBBL+MCMC to fit data.
### Jax version
The jax version can be used to calculate the gradient light curve for meachine learning or other optimization methods.For example, you can combine it with <a href='https://github.com/pyro-ppl/numpyro'> Numpyro</a> to do MCMC or HMC sampling. The combination of NUTS in numpyro and Jax version is still testing.

## Features
- [x] Optimal sampling and countour integral with error control
- [x] JIT compilation to sppeed up and GPU support
- [x] Automatic differentiation for high-performance machine learning and optimization
- [x] Laguerre's method to find the root of the polynomial and liner sum assignment algorithm to match the images

## Future work
- [ ] High order effects: parallax, orbital motion etc.
- [ ] Using NUTS in numpyro and Jax version to model real data
- [ ] More test and comparison with VBBL

## Reference
*under development* and Paper is coming soon.

