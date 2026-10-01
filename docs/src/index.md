# NoiseEstimators.jl

Robust noise floor estimation for power (i.e. amplitude squared) data.  The
[`noisefloor`](@ref) function estimates the noise floor power of data whose
noise is well modeled as the sum of two independent Gamma distributed
components with common shape `k` (for example, the sum of two
polarizations from a radio telescope), and (when `k` is known) splits the
estimated moments into the mean powers of the two components.

## Usage

```@docs
noisefloor
```

The estimate is robust to excess power contamination: where the
plain mean and standard deviation are already badly biased at a 1%
contamination fraction, `noisefloor`'s mean stays within a few percent of
the true floor even at contamination fractions of order 15%.  Arrays on an
NVIDIA GPU (e.g. `CuArray`s) are supported via the CUDA extension of
[FastQuantiles.jl](https://github.com/david-macmahon/FastQuantiles.jl),
which provides the quantile layer.

## Contents

- [Thresholding statistics](thresholding.md): `noisestats` and the
  normalization helpers for thresholding matrices of power-like data.
- [Theory of operation](theory.md): the two-Gamma noise model behind
  `noisefloor` and the `k` convention for accumulated power data.
- [Accuracy and tuning](accuracy.md): sample sizes required for the
  per-component split and choosing `qlo`.
- [Worked example](example.md): the Breakthrough Listen Voyager 2020
  single coarse channel.
