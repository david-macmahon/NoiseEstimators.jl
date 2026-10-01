# NoiseEstimators.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://david-macmahon.github.io/NoiseEstimators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://david-macmahon.github.io/NoiseEstimators.jl/dev/)
[![Build Status](https://github.com/david-macmahon/NoiseEstimators.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/david-macmahon/NoiseEstimators.jl/actions/workflows/CI.yml)

Robust noise floor estimation for power (i.e. amplitude squared) data.

The [`noisefloor`](@ref) function estimates the noise floor power of data
whose noise is well modeled as the sum of two independent Gamma
distributed components with equal shape (for example, the sum of two
polarizations from a radio telescope).  The estimate is anchored on
contamination-resistant quantiles, so it stays accurate to within a few percent
even at excess power contamination fractions of order 10%, where the plain mean
and standard deviation are already badly biased.  When the common shape
is known, the estimated moments are also split into the mean powers of the
two components.

## Usage

```julia
julia> using NoiseEstimators

julia> data = randexp(Float64, 1_000_000);   # Gamma(k=1) noise

julia> nf = noisefloor(data; k = 1)
(mean = 0.999..., std = 0.999..., shape = 1.00...,
 pow1 = 0.999..., pow2 = 1.4e-7...)

julia> nf.pow1 / nf.pow2    # second component essentially dead
7.2e5...

julia> nf.mean, nf.std
(0.999..., 0.999...)
```

Arrays on an NVIDIA GPU work the same way:

```julia
julia> using CUDA

julia> noisefloor(CuArray(data); k = 1).mean ≈ 1.0
true
```

## Installation

The package is not yet registered; install it and its FastQuantiles
dependency directly from their repositories, installing the unregistered
FastQuantiles dependency first so that it can be resolved when
NoiseEstimators is added:

```julia
julia> using Pkg

julia> Pkg.add(url = "https://github.com/david-macmahon/FastQuantiles.jl")

julia> Pkg.add(url = "https://github.com/david-macmahon/NoiseEstimators.jl")
```

## How it works (briefly)

The estimator anchors on contamination-resistant lower-tail quantiles (the
`qlo` quantile and the median), iterates an
*effective* Gamma shape to a fixed point (the two-component Gamma sum
matches an effective Gamma in its first two moments), optionally refines
the mean with a clipped-mean iteration that is bias-corrected for the
Gamma model, and (when the common shape is given) splits the
estimated moments into the mean powers of the two components.  See the
[documentation](https://david-macmahon.github.io/NoiseEstimators.jl) for
the full theory of operation.

## License

This package is licensed under the [BSD 2-Clause "Simplified"
License](LICENSE).
