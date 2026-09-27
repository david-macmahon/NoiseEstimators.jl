# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/david-macmahon/NoiseEstimators.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/david-macmahon/NoiseEstimators.jl/releases/tag/v0.1.0
