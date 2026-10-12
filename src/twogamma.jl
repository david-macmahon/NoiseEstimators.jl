# Exact two-Gamma model quantiles and the split-aware ("exact") anchoring of
# `noisefloor(...; anchor = :twogamma)`.
#
# The noise model is Gamma(k, ρθ) + Gamma(k, (1-ρ)θ) with common shape k and
# power split ρ : (1-ρ), so the unit-mean sum (θ = 1) has mean k.  The sum is
# *not* Gamma: its quantiles deviate from the moment-matched effective Gamma
# by up to tens of percent for small k and imbalanced splits (quantified by
# the paper's quantile-deviation analysis).  The default effective-Gamma
# anchoring absorbs that deviation as bias; the exact anchor instead solves
# the two-quantile system for (θ, ρ) directly: the quantile ratio
# q(qhi)/q(qlo) is scale-invariant and strictly monotone in ρ on [1/2, 1),
# so ρ follows from a 1-D root find and θ (hence mean, std, and split)
# from the level.

using SpecialFunctions: gamma_inc, gamma_inc_inv, loggamma
using LinearAlgebra: eigen, SymTridiagonal

# ------------------------------------------------------------------ CDF/quantile

# CDF of the unit-mean two-Gamma sum at `t` by Gauss-Legendre quadrature of
#     ∫ f1(x) P(k, (t-x)/θ2) dx,  f1 = pdf of Gamma(k, θ1 = ρ), θ2 = 1-ρ,
# substituting x = t·u² (regularizes the k < 1 endpoint) and doubling the
# node count until successive estimates agree to 1e-11.
const _GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
function _gl(n)
    get!(_GL_CACHE, n) do
        J = SymTridiagonal(zeros(n), [j / sqrt(4 * j^2 - 1) for j in 1:n-1])
        E = eigen(J)
        (E.values, 2 .* abs2.(E.vectors[1, :]))
    end
end
foreach(_gl, (32, 64, 128, 256))

function _twogamma_cdf_n(t, k, ρ, nodes, weights)
    θ1, θ2 = ρ, 1 - ρ
    lognorm = -loggamma(k) - k * log(θ1)
    s = 0.0
    @inbounds for i in eachindex(nodes)
        u = 0.5 * (nodes[i] + 1)             # map [-1, 1] -> [0, 1]
        x = t * u * u
        y = t - x
        G2 = y <= 0 ? 0.0 : gamma_inc(k, y / θ2)[1]
        G2 == 0 && continue
        s += weights[i] * t * u * exp((k - 1) * log(x) - x / θ1 + lognorm) * G2
    end
    s
end

function _twogamma_cdf(t, k, ρ)
    t <= 0 && return 0.0
    n = 32
    F = _twogamma_cdf_n(t, k, ρ, _gl(n)...)
    while n < 256
        n = 2n
        F2 = _twogamma_cdf_n(t, k, ρ, _gl(n)...)
        conv = abs(F2 - F) <= 1e-11
        F = F2
        conv && break
    end
    F
end

# Quantile of the unit-mean two-Gamma sum at probability p: Illinois method
# on the CDF (halving the stale bracket endpoint's value when the secant
# repeatedly lands on the same side), with a bisection-grade fallback.
function _twogamma_quantile(p, k, ρ)
    a, b = 1e-12 * k, 40 * k
    fa = _twogamma_cdf(a, k, ρ) - p
    fb = _twogamma_cdf(b, k, ρ) - p
    last = 0
    for _ in 1:45
        c = (a * fb - b * fa) / (fb - fa)
        (a < c < b) || (c = 0.5 * (a + b))
        fc = _twogamma_cdf(c, k, ρ) - p
        if fc < 0
            last == -1 && (fb *= 0.5)      # stale end halved to force movement
            a, fa = c, fc
            last = -1
        elseif fc > 0
            last == 1 && (fa *= 0.5)
            b, fb = c, fc
            last = 1
        else
            return c
        end
        b - a <= 1e-11 * max(b, 1.0) && break
    end
    0.5 * (a + b)
end

# --------------------------------------------------------------- ratio table

# Interpolants over ρ ∈ [1/2, 1] of log R(ρ), R = q(qhi)/q(qlo), and of the
# level quantile q(qhi), for the unit-mean two-Gamma sum, on Chebyshev nodes.
# One table serves every band of a batched call (k, qlo, qhi are shared);
# per-band solves are then bisections on the interpolant.
struct _TwogammaRatio
    x::Vector{Float64}                  # Chebyshev nodes on [1/2, 1]
    w::Vector{Float64}                  # barycentric weights
    y::Vector{Float64}                  # log R at the nodes
    yhi::Vector{Float64}                # q(qhi) at the nodes (unit-mean sum)
    r_min::Float64                      # R(1/2), exact (sum is Γ(2k, 1/2))
    r_max::Float64                      # R(1), exact (sum is Γ(k, 1))
    g_hi_floor::Float64                 # q(qhi) at ρ = 1/2
    g_hi_ceil::Float64                  # q(qhi) at ρ = 1
end

function _TwogammaRatio(k, qlo, qhi; n = 24)
    x = [0.75 - 0.25 * cospi((2 * j - 1) / (2 * n)) for j in 1:n]
    w = [1 / prod(x[j] - x[m] for m in 1:n if m != j) for j in 1:n]
    y = Vector{Float64}(undef, n)
    yhi = Vector{Float64}(undef, n)
    for (j, ρ) in enumerate(x)
        g_lo = _twogamma_quantile(qlo, k, ρ)
        g_hi = _twogamma_quantile(qhi, k, ρ)
        y[j] = log(g_hi / g_lo)
        yhi[j] = g_hi
    end
    r_lo(p) = 0.5 * gamma_inc_inv(2k, p, 1 - p)
    g_hi_floor = r_lo(qhi)
    g_hi_ceil = gamma_inc_inv(k, qhi, 1 - qhi)
    _TwogammaRatio(x, w, y, yhi, g_hi_floor / r_lo(qlo),
                   g_hi_ceil / gamma_inc_inv(k, qlo, 1 - qlo),
                   g_hi_floor, g_hi_ceil)
end

# Barycentric interpolation at ρ of values `y` on the table's nodes.
function _bary(t::_TwogammaRatio, y::Vector{Float64}, ρ)
    s = 0.0
    d = 0.0
    @inbounds for j in eachindex(t.x)
        ξ = ρ - t.x[j]
        ξ == 0 && return y[j]
        s += t.w[j] * y[j] / ξ
        d += t.w[j] / ξ
    end
    s / d
end

_ratio_at(t::_TwogammaRatio, ρ) = exp(_bary(t, t.y, ρ))

# Tables are shared across calls with the same (k, qlo, qhi); bounded memo.
const _TABLE_LOCK = ReentrantLock()
const _TABLES = Dict{NTuple{3,Float64},_TwogammaRatio}()
function _twogamma_ratio(k, qlo, qhi)
    lock(_TABLE_LOCK) do
        length(_TABLES) > 128 && empty!(_TABLES)
        get!(_TABLES, (Float64(k), qlo, qhi)) do
            _TwogammaRatio(k, qlo, qhi)
        end
    end
end

# ------------------------------------------------------------ exact anchoring

# Split-aware moments from two sample quantiles: ρ from the ratio (clamped to
# the identifiable range [1/2, 1]), the level quantile from the interpolant,
# then mean/std/shape from the model.  Returns (mean, std, shape, rho).
function _noise_moments_exact(qlo_val, qhi_val, t::_TwogammaRatio, k)
    R = qhi_val / qlo_val
    ρ, g_hi = if R <= t.r_min
        (0.5, t.g_hi_floor)
    elseif R >= t.r_max
        (1.0, t.g_hi_ceil)
    else
        lo, hi = 0.5, 1.0
        for _ in 1:40
            mid = 0.5 * (lo + hi)
            _ratio_at(t, mid) < R ? (lo = mid) : (hi = mid)
        end
        ρ = 0.5 * (lo + hi)
        (ρ, _bary(t, t.yhi, ρ))
    end
    c = ρ * ρ + (1 - ρ)^2
    mean_est = k * qhi_val / g_hi
    # var = k θ² c with θ = mean/k, so std = mean·√(c/k) and the effective
    # shape is mean²/std² = k/c
    (mean = mean_est, std = mean_est * sqrt(c / k), shape = k / c, rho = ρ)
end
