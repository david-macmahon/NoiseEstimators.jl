using Test
using Statistics
using Random
using NoiseEstimators

@testset "noisefloor" begin
    nrng = MersenneTwister(42)
    # One component of one integrated sample: Gamma(k, θ), i.e. the
    # sum of k unit-mean exponentials scaled by θ
    gamsamp(k, θ, n) = [θ * sum(randexp(nrng) for _ in 1:k) for _ in 1:n]
    twopol(k, θ1, θ2, n) = gamsamp(k, θ1, n) .+ gamsamp(k, θ2, n)

    # Single-component limit (θ2 = 0): exactly one Gamma
    for k in (1, 4)
        nd = twopol(k, 1.0, 0.0, 1_000_000)
        nfst = noisefloor(nd; k)
        @test nfst.mean ≈ k rtol = 0.03
        @test nfst.std ≈ sqrt(k) rtol = 0.03
        @test nfst.shape ≈ k rtol = 0.1
        @test nfst.pow1 + nfst.pow2 ≈ nfst.mean rtol = 1e-6
        @test nfst.powratio > 10  # second component essentially dead
    end

    # Two imbalanced components (θ1:θ2 = 1:2)
    nd = twopol(4, 1/3, 2/3, 1_000_000)
    nfst = noisefloor(nd; k = 4)
    @test nfst.mean ≈ 4 * (1/3 + 2/3) rtol = 0.03
    @test nfst.std ≈ sqrt(4 * (1/9 + 4/9)) rtol = 0.08
    @test 1.4 < nfst.powratio < 3.5  # true ratio 2.0 (moment-based split)
    @test nfst.pow1 > nfst.pow2 > 0

    # The lower quantile `qlo` trades contamination robustness against
    # clean-data efficiency; on clean data all reasonable values agree
    for qlo in (0.05, 0.1, 0.2)
        nfq = noisefloor(nd; k = 4, qlo)
        @test nfq.mean ≈ 4 * (1/3 + 2/3) rtol = 0.05
        @test nfq.std ≈ sqrt(4 * (1/9 + 4/9)) rtol = 0.1
    end

    # Equal components: ratio 1 and effective shape 2k.  This sits
    # exactly on the split-identification ceiling, so either a split
    # near unity or NaN (unidentified) is acceptable
    nde = twopol(4, 0.5, 0.5, 1_000_000)
    nfst = noisefloor(nde; k = 4)
    @test nfst.mean ≈ 4 rtol = 0.03
    @test isnan(nfst.powratio) || abs(nfst.powratio - 1) < 0.3
    @test nfst.shape ≈ 8 rtol = 0.1

    # Excess power contamination: 1% of bins at 100x the floor leave the
    # estimate unbiased while the plain mean is badly biased
    ndc = copy(nd)
    ndc[1:10_000] .= 100 * 4
    nfst = noisefloor(ndc; k = 4)
    @test nfst.mean ≈ 4 rtol = 0.03
    @test noisefloor(ndc; k = 4, refine = false).mean ≈ 4 rtol = 0.06
    @test mean(ndc) > 7

    # Auto-k mode: mean accurate, effective shape between k and 2k, and
    # no per-component split
    nfst = noisefloor(nd)
    @test nfst.mean ≈ 4 rtol = 0.03
    @test 4 < nfst.shape < 10
    @test nfst.powratio === nothing
    @test nfst.pow1 === nothing && nfst.pow2 === nothing

    # Data summed from Nt samples per output: per-component shape is
    # k * Nt and the floor scales accordingly (mimics a drift-rate
    # transform's path sums without depending on one)
    Nt = 16
    nds = reshape(twopol(1, 0.5, 0.5, 512 * Nt * 64), Nt, :)
    nsum = vec(sum(nds, dims = 1))
    nfst = noisefloor(nsum; k = Nt)
    @test nfst.mean ≈ Nt * (0.5 + 0.5) rtol = 0.05
    @test Nt <= nfst.shape <= 2.5 * Nt

    # Degenerate data mirrors the fallback semantics
    @test noisefloor(zeros(100)) ==
          (mean = 0.0, std = Inf, shape = nothing, powratio = nothing,
           pow1 = nothing, pow2 = nothing)
    @test noisefloor(fill(3.5, 100)) ==
          (mean = 3.5, std = Inf, shape = nothing, powratio = nothing,
           pow1 = nothing, pow2 = nothing)
    @test_throws ArgumentError noisefloor(nd; k = 0)
    @test_throws ArgumentError noisefloor(nd; qlo = 0)
    @test_throws ArgumentError noisefloor(nd; qlo = 0.5)

    # An unidentified split (data less variable than the model allows)
    # is reported as NaN rather than a spurious balanced split
    ndz = 1.0 .+ 1e-3 .* randexp(nrng, 1000)
    nfz = noisefloor(ndz; k = 4)
    @test isnan(nfz.powratio)
    @test isnan(nfz.pow1) && isnan(nfz.pow2)

    # (mean, std) projection matches the full result
    @test NoiseEstimators._noisefloor_stats(nd; k = 4) ==
          (mean = noisefloor(nd; k = 4).mean, std = noisefloor(nd; k = 4).std)
end
