## microlux: Microlensing using Jax

[![Test Status](https://github.com/coastego/microlux/actions/workflows/run_test.yml/badge.svg)](https://github.com/CoastEgo/microlux/actions/workflows/run_test.yml)
[![Documentation Status](https://github.com/coastego/microlux/actions/workflows/build_docs.yml/badge.svg)](https://coastego.github.io/microlux/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
---

`microlux` is a <a href='https://github.com/jax-ml/jax'>Jax</a>-based package that can calculate the binary lensing light curve and its derivatives both efficiently and accurately.  We use the modified adaptive contour integratoin in <a href='https://github.com/valboz/VBBinaryLensing'>`VBBinaryLensing`</a> to maximize the performance. 
With the access to the gradient, we can use more advanced algorithms for microlensing modeling, such as Hamiltonian Monte Carlo (HMC) in <a href='https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS'>`numpyro`</a>.


## Installation

The PyPI version is coming soon. Currently, You can install this package from the source code in the editable mode.
``` bash 
git clone https://github.com/CoastEgo/microlux.git
cd microlux
pip install -e .
```
## Documentation
The documentation is available at <a href='https://coastego.github.io/microlux/'>here</a>. See this for more details.


  
## Citation

`microlux` is open-source software licensed under the MIT license. If you use this package for your research, please cite our paper:

- A differentiable binary microlensing model using adaptive contour integration method: <a href='https://arxiv.org/abs/2501.07268'>in arXiv</a>