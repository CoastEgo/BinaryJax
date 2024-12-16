# Welcome to BinaryJax

BinaryJax is a Jax-based package that is used to calculate the binary lensing light curve with the finite source effect using the contour integration method.  We inherit the novel features in <a href='https://github.com/valboz/VBBinaryLensing'>VBBinaryLensing</a> including parabolic correction and optimal sampling to maximize the performance. This is built on the <a href='https://github.com/google/jax'>JAX</a> library which provides a NumPy-like interface with GPU and automatic differentiation support for high-performance machine learning research. Through automatic differentiation and our package, we get access to the accurate gradient for exploring more advanced algorithms.

## Features

- Optimal sampling and contour integration with error control to calculate the binary lens microlensing light curve with finite source effect.
- Robust and accurate calculation: Widely test over the broad parameter space compared with VBBinaryLensing
- Fast speed: Fully compatible with JIT compilation to speed up calculation. 
- Accurate gradient: Automatic differentiation with novel error estimator to ensure the convergence of gradient. 
- Aberth–Ehrlich method to find the roots of the polynomial and liner sum assignment algorithm to match the images
- Application on real events modeling using NUTS in Numpyro

## Reference
Under development and Paper is coming soon.

## Quick Start
Under development