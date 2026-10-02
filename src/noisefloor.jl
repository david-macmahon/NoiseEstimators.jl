# The noise in an integrated power spectrogram (or any other collection of
# power samples of similar provenance) is well modeled as the sum of two
# independent Gamma distributed components, `Gamma(k, θ1) + Gamma(k, θ2)`,
# with common Gamma shape `k` and possibly different mean
# powers `k * θi`.  The sum of Gammas with different scales is not itself
# Gamma, but its first two moments match an *effective* Gamma with shape
# `k_eff = mean² / std² ∈ [k, 2k]`, and the quantiles of the two-Gamma sum
# deviate from that effective Gamma's quantiles by at most a few percent
# (worst for `k = 1` with imbalanced components).  The estimator below is
# therefore anchored on signal-free quantiles interpreted through the
# effective Gamma (whose shape is iterated to a fixed point), which keeps the
# mean estimate accurate to ~1-2% and the standard deviation to ~5% in the
# worst case, both negligible for "N sigma" thresholding.  When `k` is given,
# it is used to split the estimated moments into the mean powers of the
# two components.

# qhi-quantile / mean ratio of Gamma(k, 1)
_gam_qhi_mean(k, qhi) = gamma_inc_inv(k, qhi, 1 - qhi) / k

# (qhi - qlo quantile) / standard deviation ratio of Gamma(k, 1)
_gam_qhi_qlo_sigma(k, qhi, qlo) =
    (gamma_inc_inv(k, qhi, 1 - qhi) - gamma_inc_inv(k, qlo, 1 - qlo)) / sqrt(k)

"""
    noisefloor(data; k=nothing, qlo=0.1, qhi=0.5, clip=0.0) -> NamedTuple

Estimate the noise floor power of `data` (e.g. an integrated power
spectrogram), which is modeled as the sum of two independent Gamma
distributed components with common Gamma shape `k` and possibly different
mean powers.  Genuinely single-component data (e.g. single-polarization
radio data) is handled naturally as the limit where one component's mean
power is zero,
for which the estimate is exact and the split reports `pow1 ≈ mean` and
`pow2 ≈ 0`.  The estimate is robust to excess power contamination: it is
anchored on the `qlo` and `qhi` quantiles of the data, so
the mean stays accurate to within a few percent even for contamination
fractions of order 10% (the plain mean and standard deviation break down
with far less contamination).

The returned `NamedTuple` has fields:

- `mean`: estimated noise floor power (i.e. the mean of the noise).
- `std`: estimated noise standard deviation, `sqrt(k1 * θ1^2 + k2 * θ2^2)`,
  suitable for "N sigma" thresholding.
- `shape`: the *effective* Gamma shape `mean² / std²` of the summed
  components used for the quantile conversions.  Under the model this
  lies between `k` (one component dominant) and `2k` (balanced two-
  component sum, e.g. a Stokes I radio spectrogram), since
  `1/shape = ρ² + (1 - ρ)²` for
  power ratio `ρ`.  This is estimated from the data even when `k` is
  given; a value at or above `2k` indicates data less variable than the
  model allows (e.g. after bandpass flattening).
- `pow1`, `pow2`: estimated per-component mean powers, larger component
  first.  These are only identifiable when `k` is given.  Near-balanced
  components (and small samples) yield `NaN` values, indicating that the
  split is not identified (the effective shape estimate reached the `2k`
  ceiling); `pow1 + pow2 = mean` holds only for identified splits, and
  resolving imbalances requires many samples.  See
  the [theory of operation section of the documentation](@ref
  "Theory of operation") for the model, the shape convention, split
  accuracy requirements, and a worked example.

Keyword arguments:

- `k`: Gamma shape of a single component of a data sample, and always
  the value to pass, regardless of how many components are summed into
  the data (the two-component sum is part of the model, not the `k`
  value).
  For filterbank power data this is the number of accumulated FFT frames
  per output sample (`n_accum = abs(foff) * tsamp` in Hz·s), and for data
  produced by path-summing `Nt` such samples (as done by drift-rate
  transforms), `n_accum * Nt`.  If omitted, the effective shape is
  estimated from the data and the per-component split is not reported.
- `qlo`: lower quantile paired with `qhi` for the spread estimate.
  Lower values (e.g. `0.05`) tolerate more excess power contamination; higher
  values (e.g. `0.2`) are more efficient on clean data; `0.1` is a good
  compromise.  The contamination response of `mean` is largely governed
  by `qhi` since the upper quantile anchors it.
- `qhi`: upper quantile anchoring the mean and the upper end of the
  spread estimate's span.  The default `0.5` (the median) is a good
  compromise between contamination robustness and clean-data efficiency;
  lower values tolerate more excess power contamination but make the
  spread estimate less efficient (the two anchoring quantiles become more
  correlated).  See the [choosing-the-quantiles section of the
  documentation](@ref "Choosing the quantiles") for measured tradeoffs.
- `clip`: clipping threshold, in units of the estimated mean, for the
  optional clipped-mean refinement of the mean estimate (bias-corrected
  for the Gamma model).  The default `0` disables the refinement: the
  quantile-anchored estimates are the robust choice whose accuracy does
  not depend on the distribution of the contamination.  Passing a value
  of at least 1 enables the refinement, which improves the clean-data
  efficiency of the mean (by factors of ~1.3-4 in RMS, depending on
  shape and sample size), but its contamination robustness depends on
  how the contamination is distributed in power; see the
  [choosing-the-quantiles section of the
  documentation](@ref "Choosing the quantiles").

The data is treated as a global ensemble; per-channel (bandpass) estimation
is not performed.  Degenerate data (all zeros, constant, or with a
non-positive upper quantile) yields `mean = mean(data)` and `std = Inf`
with all
other fields `nothing`.  Arrays on an NVIDIA GPU (e.g. `CuArray`s) are
supported via the CUDA extension of FastQuantiles.jl, which provides the
quantile layer.
"""
function noisefloor(data::AbstractArray{<:Real}; k=nothing, qlo=0.1, qhi=0.5,
                    clip=0.0)
    k !== nothing && k <= 0 &&
        throw(ArgumentError("k must be positive"))
    !(0 < qlo < qhi < 1) &&
        throw(ArgumentError("quantiles must satisfy 0 < qlo < qhi < 1"))
    !(clip == 0 || clip >= 1) &&
        throw(ArgumentError("clip must be 0 (no refinement) or at least 1 " *
                            "(threshold in units of the estimated mean)"))
    qlo_val, qhi_val = fast_quantile(data, [qlo, qhi])
    if !(qhi_val > 0) || !(qlo_val < qhi_val)
        return (mean = Float64(mean(data)), std = Inf, shape = nothing,
                pow1 = nothing, pow2 = nothing)
    end

    # Iterate the effective shape and mean to a fixed point: the shape
    # follows from the moment relation `k_eff = mean² / std²` and the mean
    # and standard deviation follow from the quantiles via the effective
    # Gamma conversion factors.
    mom = _noise_moments(qlo_val, qhi_val; qlo, qhi)
    mean_est = mom.mean
    std_est = mom.std
    shape = mom.shape

    if clip > 0
        # Iterated clipped mean with the exact Gamma bias correction: for
        # `X ~ Gamma(shape, θ)` with `θ = mean/shape` and clip threshold
        # `s = clip * mean`, the survivor fraction is
        # `P(shape, clip * shape)` and `E[X * 1{X < s}]` is
        # `mean * P(shape + 1, clip * shape)`, so the debiased mean is the
        # empirical survivor mean times `P(shape, c) / P(shape + 1, c)`.
        # The survivor count and sum are fused into one data pass.
        for _ in 1:5
            s = clip * mean_est
            n, total = mapreduce(x -> x < s ? (1, Float64(x)) : (0, 0.0),
                                 (a, b) -> (a[1] + b[1], a[2] + b[2]), data;
                                 init = (0, 0.0))
            n == 0 && break
            mean_new, done = _noise_refine_step(mean_est, total / n, shape, clip)
            mean_est = mean_new
            done && break
        end
    end

    if k === nothing
        pow1 = pow2 = nothing
    else
        pow1, pow2 = _noise_pow(mean_est, std_est, k)
    end

    (mean = mean_est, std = std_est, shape, pow1, pow2)
end

# Moments (mean, standard deviation, effective Gamma shape) of the
# two-Gamma noise model from signal-free quantiles, iterating the shape to a
# fixed point (the quantile-to-moment conversion factors depend on the
# shape).  Pure host math on `(qlo_val, qhi_val)`; shared by `noisefloor`
# and the batched per-band estimation of the CUDA extension.
function _noise_moments(qlo_val, qhi_val; qlo, qhi)
    shape = 2.0
    mean_est = qhi_val / _gam_qhi_mean(shape, qhi)
    std_est = 0.0
    for _ in 1:50
        std_est = (qhi_val - qlo_val) / _gam_qhi_qlo_sigma(shape, qhi, qlo)
        shape_new = clamp(mean_est^2 / std_est^2, 1e-3, 1e8)
        mean_new = qhi_val / _gam_qhi_mean(shape_new, qhi)
        done = isapprox(shape_new, shape; rtol = 1e-8)
        shape, mean_est = shape_new, mean_new
        done && break
    end
    std_est = (qhi_val - qlo_val) / _gam_qhi_qlo_sigma(shape, qhi, qlo)
    (mean = mean_est, std = std_est, shape = shape)
end

# One clipped-mean refinement step: debias the empirical survivor mean for
# the Gamma model and test convergence.  Returns the new mean estimate and
# whether it converged.
function _noise_refine_step(mean_est, empirical, shape, clip)
    p0 = gamma_inc(shape, clip * shape)[1]
    p1 = gamma_inc(shape + 1, clip * shape)[1]
    mean_new = empirical * p0 / p1
    done = isapprox(mean_new, mean_est; rtol = 1e-6)
    return mean_new, done
end

# Per-component split from the moments: `θ1 + θ2 = mean/k` and
# `θ1² + θ2² = std²/k`, so `(θ1 - θ2)² = 2 * std²/k - (mean/k)²`.  A
# non-positive value (data less variable than the model allows, e.g.
# near-balanced components with the shape estimate at the `2k` ceiling)
# leaves the split unidentified and is reported as `NaN`.  Pure host math.
function _noise_pow(mean_est, std_est, k)
    sθ = mean_est / k
    d2 = 2 * std_est^2 / k - sθ^2
    if d2 > 0
        d = sqrt(clamp(d2, 0.0, sθ^2))
        pow1 = k * (sθ + d) / 2
        pow2 = k * (sθ - d) / 2
        return pow1, pow2
    end
    return NaN, NaN
end

# (mean, std) projection of `noisefloor` (the shape that thresholding
# wrappers consume).
function _noisefloor_stats(data; k = nothing, qlo = 0.1, qhi = 0.5, clip = 0.0)
    nf = noisefloor(data; k, qlo, qhi, clip)
    (mean = nf.mean, std = nf.std)
end
