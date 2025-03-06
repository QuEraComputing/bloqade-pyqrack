# Bloqade

[![CI](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml/badge.svg)](https://github.com/QuEraComputing/bloqade/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/QuEraComputing/bloqade/graph/badge.svg?token=BpHsAYuzdo)](https://codecov.io/gh/QuEraComputing/bloqade)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/bloqade.svg?color=%2334D058)](https://pypi.org/project/bloqade)
[![Documentation](https://img.shields.io/badge/Documentation-6437FF)](https://bloqade.quera.com/)
[![DOI](https://zenodo.org/badge/629628885.svg)](https://zenodo.org/doi/10.5281/zenodo.11114109)


# Welcome to Bloqade -- QuEra's Neutral Atom SDK

Bloqade is a Python SDK for neutral atom quantum computing. It provides a set of embedded domain-specific languages (eDSLs) for programming neutral atom quantum computers. Bloqade is designed to be a high-level, user-friendly SDK that abstracts away the complexities of neutral atom quantum computing, allowing users to focus on developing quantum algorithms and compilation strategies for neutral atom quantum computers.

> [!IMPORTANT]
>
> This project is in the early stage of development. API and features are subject to change.

## Installation

This package has three different optional dependencies to decide which one, or multiple your system supports: `pyqrack`, `pyqrack-cpu`, `pyqrack-cuda`.

### Install via `uv` (Recommended)

```py
uv add bloqade-pyqrack[...]
```

### Which extra do I install??

Because how the pyqrack packages have been deployed you have to install one of the optional dependencies to get it to work on your platform depending on the status of how your platform interacts with OpenCL:

* if your on a platform that supports OpenCL but you do not have it installed you have to install `bloqade-pyqrack[pyqrack-cpu]`
* If your platform doesn't support OpenCL and you want to run it on a cpu backend you must install `bloqade-pyqrack[pyqrack]`.
* If you're system has OpenCL compatible QPU with OpenCL installed you can use your GPU via `bloqade-pyqrack[pyqrack]`
* If you have an Nvidia GPU you can install `bloqade-pyqrack[pyqrack-cuda]`.

In the future this will be simplified so that `pyqrack-cpu` will mean `cpu` only and `pyqrack` will be `cpu` and `gpu` via OpenCL and `pyqrack-cuda` will be `gpu` via CUDA.


## License

Apache License 2.0 with LLVM Exceptions
