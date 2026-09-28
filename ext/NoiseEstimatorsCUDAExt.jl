# CUDA methods for the banded noise statistics: the non-robust per-(band,
# column) reductions and the batched robust per-band estimation (whose
# per-band quantiles come from the banded `fast_quantile` of
# FastQuantiles.jl).

module NoiseEstimatorsCUDAExt

import NoiseEstimators: _banded_colstats!, _noisefloor_banded, _noise_moments,
                        _noise_refine_step
import FastQuantiles: fast_quantile, _select_eltypes

using CUDA: CuMatrix, CuDeviceMatrix, CuDeviceVector, CuArray,
            CuStaticSharedArray, @cuda, blockIdx, threadIdx, blockDim,
            gridDim, sync_threads

# One-pass banded mean/std on the device: one thread per (band, column)
# pair accumulates the band's `cpb` contiguous values in Float64 and writes
# the corrected mean and standard deviation into the preallocated outputs —
# no intermediate allocations; the matrix is read exactly once.  (A warp
# per (band, column) pair with coalesced loads and shuffle reduction was
# measured *slower* — one load in flight per warp versus one per thread —
# so this simpler mapping is kept.)  The one-pass sum/sum-of-squares
# formula differs from `Statistics.std`'s two-pass algorithm only at
# floating-point rounding level (the accumulation is done in Float64).
function _banded_colstats!(colmean::CuMatrix{T}, colstd::CuMatrix{T},
                           data::CuMatrix{T}, cpb::Int) where T
    n = length(colmean)
    if n > 0
        threads = 256
        blocks = cld(n, threads)
        @cuda threads = threads blocks = blocks _banded_colstats_kernel!(
            colmean, colstd, data, cpb)
    end
    return colmean, colstd
end

function _banded_colstats_kernel!(colmean::CuDeviceMatrix{T},
                                  colstd::CuDeviceMatrix{T},
                                  data::CuDeviceMatrix{T}, cpb::Int) where T
    lid = (Int64(blockIdx().x) - 1) * Int64(blockDim().x) + Int64(threadIdx().x)
    lid ≤ length(colmean) || return nothing
    nbands = Int(size(colmean, 1))
    b = Int((lid - 1) % nbands) + 1
    j = Int((lid - 1) ÷ nbands) + 1
    i0 = (b - 1) * cpb
    s1 = 0.0
    s2 = 0.0
    @inbounds for i in 1:cpb
        x = Float64(data[i0 + i, j])
        s1 += x
        s2 += x * x
    end
    m = s1 / cpb
    var = cpb == 1 ? 0.0 : max((s2 - s1 * m) / (cpb - 1), 0.0)
    @inbounds colmean[lid] = T(m)
    @inbounds colstd[lid] = T(sqrt(var))
    return nothing
end

"""
    _noisefloor_banded(data::CuMatrix, cpb; k, qlo, clip, refine)
        -> (mean = ..., std = ...)

CUDA method of `_noisefloor_banded`: robust per-band statistics for the
`cpb`-sized bands of rows of `data`, matching the host fallback's estimates
(the quantile selection is bit-exact; only the clipped-mean refinement sums
differ, in reduction order).  The per-band quantiles come from the batched
banded selection `FastQuantiles.fast_quantile(data, cpb, [qlo, 0.5])`, which
selects every band's quantiles in the same fixed handful of device passes;
the host-side moment estimation and the optional clipped-mean refinement
run per band from those quantiles, with the refinement's survivor count and
sum computed in fused batched count+sum passes
(`_cuda_banded_countsum`).
"""
function _noisefloor_banded(data::CuMatrix{T}, cpb::Int; k = nothing,
                            qlo = 0.1, clip = 4.0,
                            refine = true) where {T <: _select_eltypes}
    k !== nothing && k <= 0 && throw(ArgumentError("k must be positive"))
    !(0 < qlo < 0.5) && throw(ArgumentError("qlo must be between 0 and 0.5"))
    Nf = size(data, 1)
    nbands = Nf ÷ cpb
    qs = fast_quantile(data, cpb, [qlo, 0.5])
    qlo_vals = [q[1] for q in qs]
    q50s = [q[2] for q in qs]
    # Per-band moments and degenerate handling.  Degenerate bands
    # (non-positive median, or lower quantile at the median) fall back to
    # the band's plain mean and `Inf` sigma like `noisefloor`; the mean
    # comes from one batched full-range count+sum pass.  (The host
    # fallback's `Float64(mean(buf))` accumulates in the band's eltype; for
    # the zeros and constants that trigger this path both agree exactly.)
    means = Vector{Float64}(undef, nbands)
    stds = Vector{Float64}(undef, nbands)
    shapes = Vector{Float64}(undef, nbands)
    degenerate = Int[]
    for b in 1:nbands
        if !(q50s[b] > 0) || !(qlo_vals[b] < q50s[b])
            push!(degenerate, b)
            means[b] = NaN  # placeholder until the batched band mean below
            stds[b] = Inf
        else
            mom = _noise_moments(qlo_vals[b], q50s[b]; qlo)
            means[b] = mom.mean
            stds[b] = mom.std
            shapes[b] = mom.shape
        end
    end
    if !isempty(degenerate)
        thresholds = fill(Inf, length(degenerate))
        counts, sums = _cuda_banded_countsum(data, cpb, degenerate, thresholds)
        for (i, b) in enumerate(degenerate)
            means[b] = sums[i] / counts[i]
        end
    end
    if refine
        # Batched iterated clipped mean: every round thresholds each active
        # band at `clip * mean`, counts and sums the survivors in one fused
        # device pass, and advances each band independently until it
        # converges (or yields no survivors), mirroring the host loop.
        active = [b for b in 1:nbands if !(b in degenerate)]
        for _ in 1:5
            isempty(active) && break
            thresholds = [clip * means[b] for b in active]
            counts, sums = _cuda_banded_countsum(data, cpb, active, thresholds)
            still = Int[]
            for (i, b) in enumerate(active)
                n = counts[i]
                n == 0 && continue
                mean_new, done = _noise_refine_step(means[b], sums[i] / n,
                                                    shapes[b], clip)
                means[b] = mean_new
                done || push!(still, b)
            end
            active = still
        end
    end
    mean_v = Vector{Float64}(undef, Nf)
    std_v = Vector{Float64}(undef, Nf)
    for b in 1:nbands
        r = ((b - 1) * cpb + 1):(b * cpb)
        mean_v[r] .= means[b]
        std_v[r] .= stds[b]
    end
    (mean = mean_v, std = std_v)
end

# Per-band count and Float64 sum of the elements below per-band value
# thresholds, in one fused pass over each band's rows.  The comparison is
# `Float64(x) < s` so that the threshold decisions match the host's
# `x < s` (which promotes to Float64) bit for bit; NaN compares false and is
# excluded, like `count(<(s), data)`.  Each block of the fixed
# `(blocks_per_band, length(bands))` grid accumulates its own partial into a
# distinct output slot (no atomics), and the host merges the partials in
# fixed block order, so repeated calls are bitwise reproducible.
function _cuda_banded_countsum(data::CuMatrix{<:_select_eltypes}, cpb::Int,
                               bands::Vector{Int}, thresholds::Vector{Float64};
                               blocks_per_band::Int = 64)
    n_b = cpb * size(data, 2)
    nt = length(bands)
    sdev = CuArray(thresholds)
    bdev = CuArray(bands)
    counts = CuMatrix{Int64}(undef, blocks_per_band, nt)
    sums = CuMatrix{Float64}(undef, blocks_per_band, nt)
    fill!(counts, Int64(0))
    fill!(sums, 0.0)
    threads = 256
    @cuda threads = threads blocks = (blocks_per_band, nt) _kbanded_countsum!(
        counts, sums, data, sdev, bdev, cpb, n_b)
    hc = Array(counts)
    hs = Array(sums)
    out_counts = Vector{Int64}(undef, nt)
    out_sums = Vector{Float64}(undef, nt)
    for i in 1:nt
        c = Int64(0)
        s = 0.0
        for blk in 1:blocks_per_band
            c += hc[blk, i]
            s += hs[blk, i]
        end
        out_counts[i] = c
        out_sums[i] = s
    end
    return out_counts, out_sums
end

function _kbanded_countsum!(counts::CuDeviceMatrix{Int64},
                            sums::CuDeviceMatrix{Float64},
                            data::CuDeviceMatrix{<:_select_eltypes},
                            thresholds::CuDeviceVector{Float64},
                            bands::CuDeviceVector{Int}, cpb::Int, n_b::Int)
    bi = Int(blockIdx().y)
    row0 = (bands[bi] - 1) * cpb + 1
    s = thresholds[bi]
    blk = Int(blockIdx().x)
    nblk = Int(gridDim().x)
    bsz = Int(blockDim().x)
    tid = Int(threadIdx().x)
    c = Int64(0)
    acc = 0.0
    i = Int64(tid) + (Int64(blk) - 1) * Int64(bsz)
    stride = Int64(bsz) * Int64(nblk)
    while i ≤ n_b
        r = (i - 1) % cpb + 1
        col = (i - 1) ÷ cpb + 1
        @inbounds x = data[row0 + r - 1, col]
        if Float64(x) < s
            c += Int64(1)
            acc += Float64(x)
        end
        i += stride
    end
    # Reduce the threads' partials through shared memory in a fixed tree and
    # let thread 1 write the block's slot (the launcher uses a power-of-two
    # `threads` of at most 256).
    sh_c = CuStaticSharedArray(Int64, 256)
    sh_s = CuStaticSharedArray(Float64, 256)
    sh_c[tid] = c
    sh_s[tid] = acc
    sync_threads()
    h = bsz ÷ 2
    while h > 0
        if tid <= h
            sh_c[tid] += sh_c[tid + h]
            sh_s[tid] += sh_s[tid + h]
        end
        sync_threads()
        h ÷= 2
    end
    if tid == 1
        counts[blk, bi] = sh_c[1]
        sums[blk, bi] = sh_s[1]
    end
    return nothing
end

end # module NoiseEstimatorsCUDAExt
