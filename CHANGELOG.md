# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-10-01

### Changed

- Documentation: split the landing page into per-topic pages
  (thresholding statistics, theory of operation, accuracy and tuning, and
  a worked example), clarified the intro prose, and switched the README
  installation instructions from `Pkg.develop` to `Pkg.add` for
  FastQuantiles.

## [0.3.0] - 2026-09-28

### Added

- `noisestats(data; robust = true, chans_per_band, kwargs...)` computing
  the (mean, sigma) pair used for thresholding matrices of power-like
  data, robust to excess power contamination by default (anchored on
  `noisefloor` over all elements), with an iterable-of-matrices variant.
  The non-robust mode returns the plain ensemble statistics of all
  elements (pooled over each band's elements in banded mode); a constant
  matrix yields `Inf` sigma so that normalizing by it produces zeros and
  denormalizing with it produces an `Inf` threshold.
- Banded per-channel statistics via `chans_per_band`: per-band estimates
  returned as vectors of length `size(data, 1)` for per-channel
  thresholding, robust to passband power-level variations along the
  frequency axis.
- `noisenormalize`, `noisenormalize!`, and `noisedenormalize` converting
  between raw values and signal-to-noise units, with scalar or per-channel
  (vector) statistics.
- Optional CUDA extension (`NoiseEstimatorsCUDAExt`): the non-robust
  banded statistics run as a single fused one-pass kernel, and the robust
  banded statistics are computed in a fixed handful of batched device
  passes (per-band quantiles via FastQuantiles' banded `fast_quantile`,
  plus batched count+sum passes with deterministic per-block partials for
  the clipped-mean refinement), at a cost independent of the number of
  bands and bitwise reproducible across calls.

### Changed

- `noisefloor`'s estimation internals are refactored into reusable pure
  helpers (shape fixed point, clipped-mean refinement step, per-component
  split), and the clipped-mean refinement fuses its survivor count and sum
  into a single data pass.  Results are unchanged.
- `noisefloor`'s per-component estimation now runs on FastQuantiles 0.2
  (whose histogram passes reuse buffers and whose banded selection is
  shared with the CUDA extension).

## [0.2.0] - 2026-09-26

### Removed

- `noisefloor` no longer returns `powratio`; the per-component power ratio
  is simply `pow1 / pow2`.

## [0.1.0] - 2026-09-26

### Added

- `noisefloor(data; k, qlo, clip, refine)` estimating the noise floor power
  of data (e.g. an integrated power spectrogram) whose noise is modeled as
  the sum of two independent Gamma distributed components with common shape
  `k`; genuinely single-component data is handled naturally as the limiting
  case where one component's mean power is zero.
- Robustness to excess power contamination: the estimate is anchored on the
  `qlo` quantile and the median, interpreted through an *effective* Gamma
  whose shape is iterated to a fixed point, so the mean stays accurate to
  within a few percent even at contamination fractions of order 10-15%,
  where the plain mean and standard deviation are already badly biased.
- Optional clipped-mean refinement of the mean estimate with an exact Gamma
  bias correction, mainly improving statistical efficiency for small
  samples.
- Per-component split: when the common shape `k` is given, the estimated
  moments are split into the mean powers of the two components (`pow1`,
  `pow2`, `powratio`, larger component first), with `NaN` reported when the
  split is not identified.
- Support for arrays on an NVIDIA GPU (e.g. `CuArray`s) via the CUDA
  extension of FastQuantiles.jl, which provides the quantile layer.
- Documentation (theory of operation, `k` convention, split accuracy
  analysis, and a worked Voyager 2020 example), plus CI and docs-deployment
  GitHub Actions workflows.

[Unreleased]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/david-macmahon/NoiseEstimators.jl/releases/tag/v0.1.0
