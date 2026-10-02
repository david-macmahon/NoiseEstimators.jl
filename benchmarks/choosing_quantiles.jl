# Monte Carlo measurement of the (qlo, qhi) anchoring-quantile tradeoffs
# and of the opt-in clipped-mean refinement's `clip` sensitivity, as
# documented in "Choosing the quantiles" (docs/src/accuracy.md).
#
# Model: two-component Gamma noise X = Gamma(k, θ1) + Gamma(k, θ2) with
# θ1:θ2 = 70:30 (mean powers sum to the floor k), i.e. the `k = 51`
# filterbank use case.  Each trial draws N fresh samples; every estimator
# configuration sees the same clean data and the same contaminated copy,
# where a fixed fraction of samples is replaced with excess power drawn
# uniformly from 1x to 10x the floor.  Metrics (over trials):
#   clean:  RMS relative errors of mean and std, RMS split error (pp),
#           where the split error is the larger-component fraction error
#   cont.:  mean biases of mean, std, and split under contamination
#   clip:   refined-mean error/bias versus `clip` under contamination
#           (second table; the refinement is opt-in, `clip > 0`)
#
# Run with:  julia --project=. benchmarks/choosing_quantiles.jl

using Random, Statistics, Printf, NoiseEstimators

const K = 51                    # per-component Gamma shape (n_accum)
const N = 1 << 20               # samples per trial
const TRIALS = 100
const SPLIT = 0.7               # larger-component power fraction
const QLO = 0.1                 # fixed lower quantile
const QHIS = (0.3, 0.4, 0.5, 0.6, 0.7)
const CLIPS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
const CONTFRAC = 0.1            # contaminated fraction of samples
const CONTLO, CONTHI = 1.0, 10.0  # contaminant power range, in floors

rng = MersenneTwister(20260929)

# Fill `out` with Gamma(K, θ) samples via chunked sums of K exponentials
function gamfill!(out, θ, rng)
    n = length(out)
    chunk = 1 << 14
    i = 1
    while i <= n
        m = min(chunk, n - i + 1)
        @views out[i:i+m-1] .= θ .* vec(sum(randexp(rng, m, K); dims = 2))
        i += m
    end
    out
end

floor_true = Float64(K)         # θ1 + θ2 = 1
std_true = sqrt(K * (SPLIT^2 + (1 - SPLIT)^2))
g1 = Vector{Float64}(undef, N)
g2 = Vector{Float64}(undef, N)
xc = Vector{Float64}(undef, N)

# accumulators: [qhi] => (clean & contaminated) vectors per metric; the
# `anch`-suffixed metrics use the unrefined (quantile-anchored) mean,
# which is what `qhi` directly controls and which is the default path
# (`clip = 0`); `ref`-suffixed metrics use the opt-in clipped-mean
# refinement (`clip = 4`)
acc = Dict(qhi => (ref_c = Float64[], std_c = Float64[], split_c = Float64[],
                   anch_c = Float64[],
                   ref_f = Float64[], std_f = Float64[], split_f = Float64[],
                   anch_f = Float64[],
                   unid_c = 0, unid_f = 0) for qhi in QHIS)
clipacc = Dict(c => (err_c = Float64[], bias_f = Float64[]) for c in CLIPS)

for t in 1:TRIALS
    gamfill!(g1, SPLIT, rng)
    gamfill!(g2, 1 - SPLIT, rng)
    x = g1 .+ g2
    ncont = round(Int, CONTFRAC * N)
    copyto!(xc, x)
    @views xc[1:ncont] .= floor_true .* (CONTLO .+ (CONTHI - CONTLO) .* rand(rng, ncont))
    for qhi in QHIS
        a = acc[qhi]
        for (data, tag) in ((x, :clean), (xc, :cont))
            nf = noisefloor(data; k = K, qlo = QLO, qhi, clip = 4.0)
            nq = noisefloor(data; k = K, qlo = QLO, qhi)
            frac = isnan(nq.pow1) ? NaN : nq.pow1 / (nq.pow1 + nq.pow2)
            rerr = 100 * (nf.mean / floor_true - 1)
            aerr = 100 * (nq.mean / floor_true - 1)
            serr = 100 * (nf.std / std_true - 1)
            sperr = 100 * (frac - SPLIT)
            if tag === :clean
                push!(a.ref_c, rerr); push!(a.std_c, serr); push!(a.split_c, sperr)
                push!(a.anch_c, aerr)
                isnan(frac) && (a.unid_c += 1)
            else
                push!(a.ref_f, rerr); push!(a.std_f, serr); push!(a.split_f, sperr)
                push!(a.anch_f, aerr)
                isnan(frac) && (a.unid_f += 1)
            end
        end
    end
    for c in CLIPS
        push!(clipacc[c].err_c, 100 * (noisefloor(x; k = K, clip = c).mean / floor_true - 1))
        push!(clipacc[c].bias_f, 100 * (noisefloor(xc; k = K, clip = c).mean / floor_true - 1))
    end
end

rms(v) = sqrt(mean(abs2, filter(isfinite, v)))
bias(v) = mean(filter(isfinite, v))

println("k = $K, N = $N, trials = $TRIALS, split = $(SPLIT)/$(1 - SPLIT), ",
        "qlo = $QLO; contamination: $(round(Int, 100 * CONTFRAC))% of samples ",
        "at Uniform($(CONTLO), $(CONTHI))x floor")
println()
println("| qhi | refined mean RMS% | std RMS% | split RMS pp | anchored mean RMS% | refined mean bias% | std bias% | split bias pp | anchored mean bias% |")
println("|-----|-------------------|----------|--------------|--------------------|--------------------|-----------|---------------|---------------------|")
for qhi in QHIS
    a = acc[qhi]
    @printf("| %.1f | %6.2f | %6.2f | %6.2f | %6.2f | %+6.2f | %+6.2f | %+6.2f | %+6.2f |\n",
            qhi, rms(a.ref_c), rms(a.std_c), rms(a.split_c), rms(a.anch_c),
            bias(a.ref_f), bias(a.std_f), bias(a.split_f), bias(a.anch_f))
end
println()
println("Refined mean (opt-in, `clip > 0`), clean error and contaminated bias versus clip:")
println()
println("| clip | clean mean err% | contaminated mean bias% |")
println("|------|-----------------|-------------------------|")
for c in CLIPS
    @printf("| %.1f | %+6.2f | %+7.2f |\n", c, bias(clipacc[c].err_c), bias(clipacc[c].bias_f))
end
println()
plain = 100 * (mean(xc) / floor_true - 1)
@printf("plain mean bias: %+.1f%%\n", plain)
