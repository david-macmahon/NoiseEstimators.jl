# Thresholding statistics for power-like matrices: scalar or per-band
# (banded) estimates of the noise floor's mean and standard deviation, plus
# the matching normalize/denormalize helpers.  The robust estimates are
# anchored on `noisefloor`'s quantile-based model; the plain estimates are
# the classical moment-based statistics of the first non-degenerate column.

# A few divisors of `Nf` nearest to `cpb`, for the divisibility error
# message.
function _nearest_divisors(Nf::Int, cpb::Int)
    out = Int[]
    for d in 0:max(cpb, Nf)
        for c in (d == 0 ? (cpb,) : (cpb - d, cpb + d))
            if 1 <= c <= Nf && Nf % c == 0 && c ∉ out
                push!(out, c)
                length(out) == 3 && return out
            end
        end
    end
    return out
end

"""
    noisestats(data; robust=true, chans_per_band=nothing, kwargs...)
        -> (mean = ..., std = ...)
    noisestats(datas; robust=true, kwargs...) -> (mean = ..., std = ...)

Compute the mean and standard deviation (aka sigma) used for thresholding
the matrix `data` (or matrices `datas`) of power-like values — e.g. a
frequency-by-drift-rate or frequency-by-time matrix.  If `robust` is true
(the default), the statistics are estimated over all elements of `data`
with [`noisefloor`](@ref), which is robust to excess power contamination
(see its docstring); for an iterable, the mean comes from the first matrix
with finite sigma and the sigma is the minimum across matrices.  If
`robust` is false, the plain statistics are returned instead: the mean and
standard deviation of all elements of `data` (of each matrix of `datas`,
combined the same way as for the robust mode).  A constant matrix
(including all zeros) has zero standard deviation, which is reported as
`Inf`, so that normalizing by it produces zeros and denormalizing with it
produces an `Inf` threshold (i.e. no hits).

The optional `k` keyword (per-component Gamma shape) is passed through to
`noisefloor` and improves the accuracy of the standard deviation estimate
when the integration factor of the data is known.  The `qlo`, `clip`, and
`refine` keywords of `noisefloor` are also passed through (they are
ignored when `robust` is false).

When the integer `chans_per_band` is given, the statistics are instead
estimated per *band* of `chans_per_band` rows (each band's statistics pool
its `chans_per_band`-by-`Nr` block of values) and returned as vectors of
length `size(data, 1)`, with each band's estimate repeated for every
channel of the band.  This enables per-channel thresholding while pooling
enough samples per band for well-sampled estimates; `chans_per_band = 1`
estimates each row independently.  `chans_per_band` must evenly divide the
number of rows.  Banding is not supported for iterables of matrices.

Banding matters when the noise power varies along the frequency axis, e.g.
from power-level variations across a coarse channel's passband that survive
the analytic filter-response correction: one estimate for the whole channel
is then biased wherever the power differs, while per-band estimates track
the variation.  Pick the band width small enough that the variation within
a band is negligible, and large enough to pool many samples (e.g. 16K
channels, ≈ 48 kHz for a ~2.9 Hz channelization).  On CUDA, the robust
per-band estimates are computed in a fixed handful of batched device
passes, at a cost independent of the number of bands; on the host, each
band is estimated independently.

The returned values are a `NamedTuple` with fields `mean` and `std`, so both
`m, s = noisestats(data)` and `noisestats(data).std` work (scalars, or vectors
of length `size(data, 1)` when `chans_per_band` is given).
"""
function noisestats(data::AbstractMatrix; robust::Bool = true,
                    chans_per_band::Union{Nothing, Integer} = nothing,
                    kwargs...)
    if chans_per_band !== nothing
        cpb = Int(chans_per_band)
        cpb >= 1 ||
            throw(ArgumentError("chans_per_band must be at least 1 (got $cpb)"))
        Nf = size(data, 1)
        Nf % cpb == 0 || throw(ArgumentError(
            "chans_per_band (= $cpb) must evenly divide the number of " *
            "channels ($Nf); nearest: " *
            join(_nearest_divisors(Nf, cpb), ", ")))
        return _banded_stats(data, cpb; robust, kwargs...)
    end
    if robust
        return _noisefloor_stats(data; kwargs...)
    end
    m = mean(data)
    s = std(data)
    s = s == 0 ? oftype(s, Inf) : s
    (mean = m, std = s)
end

function noisestats(datas; robust::Bool = true, chans_per_band = nothing,
                    kwargs...)
    chans_per_band === nothing || throw(ArgumentError(
        "chans_per_band is only supported for a single matrix"))
    stats = [noisestats(data; robust, kwargs...) for data in datas]
    s = minimum(st.std for st in stats; init=Inf)
    i = findfirst(st -> st.std < Inf, stats)
    m = i === nothing ? first(stats).mean : stats[i].mean
    (mean = m, std = s)
end

# Per-band (mean, std) statistics for `chans_per_band`-sized bands of
# channels (`cpb` is guaranteed to evenly divide the channel count),
# returned as vectors of length `size(data, 1)` with each band's estimate
# repeated over the channels of the band.
#
# Robust (quantile-based) statistics are not reductions; they are delegated
# to `_noisefloor_banded`, which the CUDA extension implements as a batched
# device computation whose cost is independent of the number of bands.  The
# host fallback below estimates each band independently.  Non-robust
# statistics are plain reductions computed directly on the matrix by
# `_banded_colstats!` — no bands are materialized (a single fused pass on
# CUDA).
function _banded_stats(data::AbstractMatrix, cpb::Int; robust::Bool, kwargs...)
    if !robust
        return _banded_stats_reduce(data, cpb)
    end
    return _noisefloor_banded(data, cpb; kwargs...)
end

# Host fallback for `_noisefloor_banded`: robust per-band statistics via an
# independent `noisefloor` estimate per band.  Each band is broadcast into a
# single reusable buffer, so repeated calls do not churn a full-matrix worth
# of allocations.  (The buffer also sidesteps `std` of a 2D SubArray, which
# falls back to scalar iteration and is disallowed for GPU arrays.)
function _noisefloor_banded(data::AbstractMatrix, cpb::Int; kwargs...)
    Nf, Nr = size(data)
    nbands = Nf ÷ cpb
    buf = similar(data, cpb, Nr)
    bstats = map(1:nbands) do bi
        r = ((bi - 1) * cpb + 1):(bi * cpb)
        buf .= @view data[r, :]
        _noisefloor_stats(buf; kwargs...)
    end
    means = [st.mean for st in bstats]
    stds = [st.std for st in bstats]
    T = promote_type(eltype(means), eltype(stds))
    mean_v = Vector{T}(undef, Nf)
    std_v = Vector{T}(undef, Nf)
    for (bi, st) in enumerate(bstats)
        r = ((bi - 1) * cpb + 1):(bi * cpb)
        mean_v[r] .= st.mean
        std_v[r] .= st.std
    end
    (mean = mean_v, std = std_v)
end

# Per-(band, column) means and (corrected) standard deviations of the
# `cpb`-sized bands of `data`'s rows, written into the preallocated
# `(Nf ÷ cpb, Nr)` outputs.  Host fallback: reshape reductions (see
# `_banded_stats` for why the bands are not viewed).
function _banded_colstats!(colmean::AbstractMatrix, colstd::AbstractMatrix,
                           data::AbstractMatrix, cpb::Int)
    nbands = size(colmean, 1)
    R = reshape(data, cpb, nbands, size(data, 2))
    colmean .= dropdims(mean(R, dims=1), dims=1)
    colstd .= dropdims(std(R, dims=1), dims=1)
    return colmean, colstd
end

# Non-robust banded statistics: the plain statistics of each band's
# `cpb`-by-`Nr` block of values, pooled on the host from the per-(band,
# column) means and standard deviations of `_banded_colstats!` by exact
# moment combination (all columns of a band have the same count `cpb`):
# with `M` the mean of the column means, the pooled corrected variance is
# `((cpb - 1) * Σs_c² + cpb * Σ(m_c - M)²) / (cpb * Nr - 1)`.
function _banded_stats_reduce(data::AbstractMatrix, cpb::Int)
    Nf, Nr = size(data)
    nbands = Nf ÷ cpb
    colmean = similar(data, nbands, Nr)
    colstd = similar(data, nbands, Nr)
    _banded_colstats!(colmean, colstd, data, cpb)
    hmean = Array(colmean)
    hstd = Array(colstd)               # small (nbands, Nr) download
    T = promote_type(eltype(hmean), eltype(hstd))
    mean_v = Vector{T}(undef, Nf)
    std_v = Vector{T}(undef, Nf)
    for b in 1:nbands
        m_c = Float64.(view(hmean, b, :))
        s_c = Float64.(view(hstd, b, :))
        M = mean(m_c)
        var = ((cpb - 1) * sum(abs2, s_c) +
               cpb * sum(x -> abs2(x - M), m_c)) / (cpb * Nr - 1)
        s = if var > 0
            sqrt(var)
        elseif var == 0
            Inf
        else
            var  # NaN propagates
        end
        r = ((b - 1) * cpb + 1):(b * cpb)
        mean_v[r] .= T(M)
        std_v[r] .= T(s)
    end
    (mean = mean_v, std = std_v)
end

"""
    noisenormalize(scalar, m, s) -> normalized_scalar

Normalize the value `scalar` by subtracting the mean `m` and dividing by
the standard deviation `s`.  See [`noisenormalize!`](@ref) for in-place
normalization of matrices and [`noisestats`](@ref) for computing `m` and
`s` (which is robust to excess power contamination by default).
"""
function noisenormalize(value::Number, m, s)
    value = (value - m) / s
    return value
end

"""
    noisenormalize!(data[, m, s]) -> data (normalized in place)
    noisenormalize!(datas[, m, s]) -> datas (normalized in place)

Normalize the matrix `data` (or matrices `datas`) in place by subtracting
the mean and dividing by the standard deviation.  If not given, the
statistics are computed with [`noisestats`](@ref) (which is robust to
excess power contamination by default).  The mean and standard deviation
may also be given explicitly as `m` and `s`, respectively — as scalars, or
as equal-length vectors for per-channel normalization (which broadcasts
along the frequency axis).
"""
function noisenormalize!(data::AbstractMatrix, m, s)
    data .= (data .- m) ./ s
    return data
end

function noisenormalize!(data::AbstractMatrix)
    m, s = noisestats(data)
    return noisenormalize!(data, m, s)
end

function noisenormalize!(datas, m, s)
    for data in datas
        data .= (data .- m) ./ s
    end
    return datas
end

function noisenormalize!(datas)
    m, s = noisestats(datas)
    return noisenormalize!(datas, m, s)
end

"""
    noisedenormalize(snr, m, s) -> threshold
    noisedenormalize(snr, data) -> threshold
    noisedenormalize(snr, datas) -> threshold

Compute the denormalized value of `snr` using the mean `m` and standard
deviation `s`.  If a matrix `data` (or matrices `datas`) is passed instead
of `m` and `s` the statistics will be computed from the given data with
[`noisestats`](@ref) (which is robust to excess power contamination by
default).  The denormalized value can be used as the threshold when
detecting above-threshold points in `data` rather than normalizing `data`
and using the `snr` value directly.  `m` and `s` may be equal-length
vectors, in which case a vector of per-channel thresholds is returned.
"""
function noisedenormalize(snr, m, s)
    threshold = snr * s + m
    return threshold
end

function noisedenormalize(snr, data::AbstractMatrix)
    m, s = noisestats(data)
    return noisedenormalize(snr, m, s)
end

function noisedenormalize(snr, datas)
    m, s = noisestats(datas)
    return noisedenormalize(snr, m, s)
end
