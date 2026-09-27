# NoiseEstimators.jl

Robust noise floor estimation for power (i.e. amplitude squared) data.  The
[`noisefloor`](@ref) function estimates the noise floor power of data whose
noise is well modeled as the sum of two independent Gamma distributed
components with common shape `k` (the natural distribution of integrated
power samples from a radio spectrometer, where the two components are the
polarizations), and (when `k` is known) splits the estimated moments into
the mean powers of the two components.

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

## Theory of operation

`noisefloor` has three parts: an *initial iteration* that produces the
quantile-anchored estimates, an optional *refinement* of the mean, and the
*per-component split*.  All three rest on the same model: the noise is
the sum of two independent Gamma distributed components, `Gamma(k, θ1) +
Gamma(k, θ2)`, with common shape `k` and possibly different
scales `θi` (mean powers `k·θi`).

### Initial iteration (quantile anchoring + effective shape)

The two-Gamma sum is not itself Gamma, but its first two moments match an
*effective* Gamma with shape `k_eff = mean²/std² = k/(ρ² + (1−ρ)²)` for
power fraction `ρ = θ1/(θ1+θ2)`, which lies in `[k, 2k]` (equal to `k` when
one component dominates and `2k` for a balanced two-component sum, e.g.
Stokes I in a radio spectrogram).  The
quantiles of the two-Gamma sum deviate from that effective Gamma's
quantiles by at most a few percent (worst for `k = 1` with imbalanced
components), so the data's quantiles can be interpreted through an
effective Gamma whose shape is itself estimated from the data:

1. Compute the signal-free quantiles `qlo` (default 10%) and the median
   with [`FastQuantiles.fast_quantile`](https://david-macmahon.github.io/FastQuantiles.jl).
2. Start from an initial effective shape and iterate to a fixed point:
   the mean follows from the median via the Gamma `median/mean` ratio
   `gam_med_mean(s) = Γ⁻¹(s, 1/2)/s`; the standard deviation follows from
   the quantile span via `gam_med_qlo_sigma(s, qlo)`; and the new shape is
   `mean²/std²` (clamped to `[1e-3, 1e8]`).  Each pass's shape feeds the
   next pass's conversion factors until `shape` stops moving (rtol 1e-8,
   at most 50 iterations).

Because the anchoring quantiles sit in the lower, signal-free tail of the
distribution (where contamination, which only adds power, has little
influence), the resulting `mean` and `std` remain accurate under
contamination that catastrophically biases plain moments.  The median
anchors the mean so its contamination response is largely independent of
`qlo`; the spread estimate trades robustness against efficiency through
`qlo` (see [Choosing `qlo`](@ref)).

### Refinement (clipped mean, optional)

With `refine = true` (the default), the mean is refined by an iterated
clipped mean with an *exact* Gamma bias correction.  For `X ~ Gamma(shape,
θ)` with clip threshold `s = clip·mean`, the survivor fraction is
`P(shape, clip·shape)` (regularized lower incomplete gamma) and the
survivor mean is `E[X·1{X<s}] = mean·P(shape+1, clip·shape)`, so the
debiased mean estimate is

```math
\widehat{mean} = \underbrace{\frac{\sum_{x_i < s} x_i}{\#\{x_i < s\}}}_{\text{empirical survivor mean}}
                \cdot \frac{P(\text{shape}, c)}{P(\text{shape}+1, c)},
\qquad c = \text{clip}\cdot\text{shape}.
```

Iterating (at most 5 times, rtol 1e-6) converges the clipped threshold and
the mean simultaneously.  The refinement touches only the **mean**: the
`std` (and hence the split) remains the quantile-derived estimate.  Since
the quantile-anchored mean is already unbiased, the refinement mainly
improves *statistical efficiency* for small samples.

### Per-component split

When `k` is given, the moments are split into per-component scales via
`θ1 + θ2 = mean/k` and `θ1² + θ2² = std²/k`, i.e.
`(θ1 − θ2)² = 2·std²/k − (mean/k)²`.  A positive value gives
`pow1,2 = k(sθ ± √d2)/2` (larger first); a
non-positive value (data less variable than the model allows, which
happens near balanced components where the effective shape estimate
reaches the `2k` ceiling) leaves the split *unidentified*, reported as
`NaN` rather than a spurious balanced split.  The split is a small
difference of large quantities, so its accuracy degrades steeply toward
balanced components; the [sample-size requirements](@ref
"Per-component split accuracy") are quantified below.

## The `k` convention

`k` is the Gamma *shape* of a single component of a data sample, and that
is always the value to pass, regardless of how many components are
summed into the data (the two-component sum is part of the model, not
the `k` value).  A spectrometer output sample is the sum of `n_accum`
accumulated FFT frames, and each frame's `real^2 + imag^2` power
contributes 2 degrees of freedom (i.e. Gamma shape 1) per polarization, so
for filterbank power data

```julia
k = n_accum = abs(foff) * tsamp   # foff in Hz, tsamp in s
```

and for data produced by path-summing `Nt` such samples (e.g. a
Frequency-Drift-Rate matrix), `k = n_accum * Nt`.  The number of summed
components does *not* enter `k`; it only shows up in the *reported*
effective `shape`, which is `k` when one component dominates and
approaches `2k` for a balanced two-component sum.  Genuinely single-
component data needs no special treatment either: it is the limit where
one component's mean power is zero, for which the estimate is exact and the
split reports `pow1 ≈ mean` and `pow2 ≈ 0`.

As a concrete instance, the Breakthrough Listen Voyager 2020 single coarse
channel file has `foff = 2.794 Hz` and `tsamp = 18.2536 s`, so
`n_accum = 51`; pass `k = 51` for the spectrogram and `k = 51 * 16 = 816`
for its drift-rate path sums, whether using single-polarization or
Stokes I data.

## Per-component split accuracy

The split is computed from the estimated moments via
`(θ1 - θ2)² = 2·std²/k - (mean/k)²`, so its accuracy is limited by the
standard deviation estimate and degrades steeply toward balanced
components (a small difference of large quantities).  The table below
gives the approximate number of samples required for the
larger-component fraction to be accurate to within the stated number of
percentage points (RMS; Monte Carlo measured at `k = 51` for pure Gamma
noise):

| split | N @ 2 pp | N @ 5 pp | N @ 10 pp | N @ 20 pp |
|-------|----------|----------|-----------|-----------|
| 100/0 | 2.9e3    | 4.6e2    | 1.2e2     | 2.9e1     |
| 90/10 | 6.6e3    | 9.9e2    | 2.5e2     | 6.1e1     |
| 80/20 | 1.0e4    | 1.4e3    | 3.3e2     | 8.3e1     |
| 70/30 | 1.3e4    | 1.8e3    | 4.5e2     | 1.1e2     |
| 60/40 | 3.9e4    | 6.0e3    | 1.5e3     | 3.7e2     |
| 50/50 | never¹   | 7.0e4    | 1.5e4     | 3.6e3     |

¹ A bias floor of about 2 pp (about 1 pp for imbalanced splits) does not
average down, so near-balanced splits are systematically limited; at exactly
50/50 roughly half of the runs report an unidentified split (`NaN`).  The
values are Monte Carlo estimates, good to ~30%.

## Choosing `qlo`

The `qlo` keyword (lower quantile paired with the median) trades
contamination robustness against clean-data efficiency: lower quantiles are
less affected by excess power contamination (the contaminated samples sit in
the upper tail, and the quantile-value shift under contamination is
smallest where the Gamma density is steepest), while higher quantiles give
a more efficient spread estimate on clean data (the quantile correlation
with the median grows with the quantile index).  Monte Carlo at `k = 51`
gives a clean-data split error of 0.68/0.61/0.37 pp for `qlo` of
0.05/0.1/0.2, versus a split bias of +4.3/+5.3/+6.8 pp at a 10%
contamination fraction; the `mean` estimate is largely `qlo`-independent
since the median anchors it.  The default `qlo = 0.1` is a good compromise;
since contamination levels can vary widely, `qlo` can be
tuned per search.

## Worked example: Voyager 2020 single coarse channel

The bandpass of that file is flat (coefficient of variation ~3%) across the
central 80% of its 2^20 channels and rolls off toward the band edges (the
-3 dB point is essentially at the edge), so the central 80% needs no
flattening.  Running `noisefloor` with `k = 51` on the central 80% of the
band (about 13.4 million samples, unflattened) recovers a polarization
split of 64.3%/35.7%, in agreement with the 62%/38% measured directly from
the full-polarization version of the same observation; the residual ~2 pp
is real-data systematics (residual bandpass and RFI) on top of the ~0.6 pp
statistical error at this sample size.
