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
        @test nfst.pow1 / nfst.pow2 > 10  # second component essentially dead
    end

    # Two imbalanced components (θ1:θ2 = 1:2)
    nd = twopol(4, 1/3, 2/3, 1_000_000)
    nfst = noisefloor(nd; k = 4)
    @test nfst.mean ≈ 4 * (1/3 + 2/3) rtol = 0.03
    @test nfst.std ≈ sqrt(4 * (1/9 + 4/9)) rtol = 0.08
    @test 1.4 < nfst.pow1 / nfst.pow2 < 3.5  # true ratio 2.0 (moment-based split)
    @test nfst.pow1 > nfst.pow2 > 0

    # The lower quantile `qlo` trades contamination robustness against
    # clean-data efficiency; on clean data all reasonable values agree
    for qlo in (0.05, 0.1, 0.2)
        nfq = noisefloor(nd; k = 4, qlo)
        @test nfq.mean ≈ 4 * (1/3 + 2/3) rtol = 0.05
        @test nfq.std ≈ sqrt(4 * (1/9 + 4/9)) rtol = 0.1
    end

    # The upper quantile `qhi` anchors the mean and the upper end of the
    # spread's span; on clean data all reasonable values agree (the
    # default 0.5 is the median)
    for qhi in (0.3, 0.4, 0.5, 0.6)
        nfh = noisefloor(nd; k = 4, qhi)
        @test nfh.mean ≈ 4 * (1/3 + 2/3) rtol = 0.05
        @test nfh.std ≈ sqrt(4 * (1/9 + 4/9)) rtol = 0.1
    end

    # The clipped-mean refinement (opt-in via clip > 0) improves the
    # clean-data mean
    @test noisefloor(nd; k = 4, clip = 4).mean ≈ 4 * (1/3 + 2/3) rtol = 0.03

    # Equal components: ratio 1 and effective shape 2k.  This sits
    # exactly on the split-identification ceiling, so either a split
    # near unity or NaN (unidentified) is acceptable
    nde = twopol(4, 0.5, 0.5, 1_000_000)
    nfst = noisefloor(nde; k = 4)
    @test nfst.mean ≈ 4 rtol = 0.03
    @test isnan(nfst.pow1) || abs(nfst.pow1 / nfst.pow2 - 1) < 0.3
    @test nfst.shape ≈ 8 rtol = 0.1

    # Excess power contamination: 1% of bins at 100x the floor leave the
    # estimate unbiased while the plain mean is badly biased
    ndc = copy(nd)
    ndc[1:10_000] .= 100 * 4
    nfst = noisefloor(ndc; k = 4)
    @test nfst.mean ≈ 4 rtol = 0.03
    # The opt-in clipped-mean refinement also stays accurate for these
    # rare strong outliers
    @test noisefloor(ndc; k = 4, clip = 4).mean ≈ 4 rtol = 0.06
    # A lower upper-quantile keeps (and slightly improves) the robustness
    @test noisefloor(ndc; k = 4, qhi = 0.4).mean ≈ 4 rtol = 0.03
    @test mean(ndc) > 7

    # Auto-k mode: mean accurate, effective shape between k and 2k, and
    # no per-component split
    nfst = noisefloor(nd)
    @test nfst.mean ≈ 4 rtol = 0.03
    @test 4 < nfst.shape < 10
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
          (mean = 0.0, std = Inf, shape = nothing,
           pow1 = nothing, pow2 = nothing)
    @test noisefloor(fill(3.5, 100)) ==
          (mean = 3.5, std = Inf, shape = nothing,
           pow1 = nothing, pow2 = nothing)
    @test_throws ArgumentError noisefloor(nd; k = 0)
    @test_throws ArgumentError noisefloor(nd; qlo = 0)
    @test_throws ArgumentError noisefloor(nd; qlo = 0.5)
    @test_throws ArgumentError noisefloor(nd; qhi = 1.0)
    @test_throws ArgumentError noisefloor(nd; qhi = 0.05)  # qhi below qlo
    @test_throws ArgumentError noisefloor(nd; qhi = 0.5, qlo = 0.5)
    @test_throws ArgumentError noisefloor(nd; clip = 0.5)  # in (0, 1)
    @test_throws ArgumentError noisefloor(nd; clip = -1)

    # An unidentified split (data less variable than the model allows)
    # is reported as NaN rather than a spurious balanced split
    ndz = 1.0 .+ 1e-3 .* randexp(nrng, 1000)
    nfz = noisefloor(ndz; k = 4)
    @test isnan(nfz.pow1) && isnan(nfz.pow2)

    # (mean, std) projection matches the full result
    @test NoiseEstimators._noisefloor_stats(nd; k = 4) ==
          (mean = noisefloor(nd; k = 4).mean, std = noisefloor(nd; k = 4).std)
end

@testset "noisestats" begin
    # The plain (robust = false) mode is the ensemble statistics of all
    # elements; a constant matrix (including all zeros) has zero standard
    # deviation, reported as Inf so that normalizing by it produces zeros
    # and denormalizing with it produces an Inf threshold
    # (noisenormalize! used to divide by zero, producing NaNs)
    rngs = MersenneTwister(3)
    data0 = randexp(rngs, Float32, 37, 64)
    data1 = zeros(Float32, 37, 64)              # constant (all zero)
    m, s = noisestats(data0; robust = false)
    @test m == mean(data0)
    @test s == std(data0) > 0
    @test noisestats(data0; robust = false).std == s  # named access
    @test noisestats([data0, data1]; robust = false) == (mean = m, std = s)
    data = copy(data0)
    @test all(isfinite, noisenormalize!(data))

    @test noisestats(data1; robust = false) == (mean = 0.0f0, std = Inf)
    @test noisenormalize!(data1) == zeros(37, 64)
    @test noisedenormalize(5.0, data1) == Inf

    # Iterable-of-matrices variants: the mean comes from the first matrix
    # with finite sigma and the sigma is the minimum across matrices
    @test noisestats([data1]; robust = false) == (mean = 0.0f0, std = Inf)
    datas = [copy(data1), copy(data0)]
    noisenormalize!(datas)
    @test all(isfinite, datas[1]) && all(isfinite, datas[2])
    @test noisedenormalize(5.0, [copy(data1)]) == Inf
    @test noisestats([copy(data1), data0]; robust = false) == (mean = m, std = s)
    @test noisestats(fill(3.0f0, 4, 8); robust = false) ==
          (mean = 3.0f0, std = Inf)

    # Robust estimation (delegates to noisefloor): excess power
    # contamination leaves (mean, std) stable while the plain statistics
    # are badly biased.  The data is Gamma(2, 1) per sample (two unit-mean
    # exponential components with k = 2).
    rngf = MersenneTwister(7)
    gm = randexp(rngf, Float32, 256, 256) .+ randexp(rngf, Float32, 256, 256)
    gc = copy(gm)
    gc[1:1000] .= 1000f0
    mr, sr = noisestats(gc; robust = true, k = 2)
    @test mr ≈ 2 rtol = 0.05
    @test sr ≈ sqrt(2) rtol = 0.15
    @test noisestats(gc; robust = true, k = 2, qlo = 0.2).mean ≈ 2 rtol = 0.05
    @test noisestats(gc; robust = false).mean > 5
    @test noisestats(fill(3.0f0, 4, 8); robust = true) == (mean = 3.0, std = Inf)
    @test noisestats(zeros(Float32, 4, 4); robust = true) == (mean = 0.0, std = Inf)
    @test noisestats([gc, gc]; robust = true, k = 2) ==
          noisestats(gc; robust = true, k = 2)
    # robust is the default for noisestats; contamination inflates the
    # plain mean but leaves the robust mean accurate
    @test noisestats(gc).mean ≈ 2 rtol = 0.05
    @test noisestats(gc; robust = false).mean > 5
end

@testset "noisestats per-channel" begin
    rngb = MersenneTwister(7)
    # Two 8-channel bands of Gamma(1, 1) noise (exponential power, the
    # domain of the robust estimator); 2048 samples per band keeps the
    # anchored (default, no refinement) estimates well within tolerance
    datab = zeros(16, 256)
    datab[1:8, :] .= randexp(rngb, 8, 256)
    datab[9:16, :] .= 4.0 .* randexp(rngb, 8, 256)

    # Banded statistics repeat each band's estimate over its channels; the
    # plain mode pools each band's 8-by-64 block of 512 values, so its
    # estimates are already well sampled (exponential noise: sigma = mean)
    st = noisestats(datab; chans_per_band = 8, robust = false)
    @test length(st.mean) == 16 && length(st.std) == 16
    @test st.mean[1] ≈ 1 rtol = 0.15
    @test st.mean[9] ≈ 4 rtol = 0.15
    @test st.mean[1:8] == fill(st.mean[1], 8)
    @test st.mean[9:16] == fill(st.mean[9], 8)
    @test st.std[1:8] == fill(st.std[1], 8)
    @test st.std[9:16] == fill(st.std[9], 8)
    @test st.std[1] ≈ 1 rtol = 0.15
    @test st.std[9] ≈ 4 rtol = 0.15

    # Robust banded statistics recover the band noise floors
    str = noisestats(datab; chans_per_band = 8)
    @test str.mean[1] ≈ 1 rtol = 0.1
    @test str.mean[9] ≈ 4 rtol = 0.1
    @test str.std[1] ≈ 1 rtol = 0.25
    @test str.std[9] ≈ 4 rtol = 0.25

    # Per-channel estimation (chans_per_band = 1)
    st1 = noisestats(datab; chans_per_band = 1)
    @test length(st1.mean) == 16
    @test st1.mean[3] ≈ 1 rtol = 0.4
    @test st1.mean[12] ≈ 4 rtol = 0.4

    # Degenerate band (all-zero rows): Inf sigma, like the scalar path
    dataz = zeros(8, 16)
    dataz[5:8, :] .= 3.0 .+ randn(rngb, 4, 16)
    stz = noisestats(dataz; chans_per_band = 4, robust = false)
    @test stz.mean[1:4] == [0.0, 0.0, 0.0, 0.0]
    @test stz.std[1:4] == [Inf, Inf, Inf, Inf]
    @test isfinite(stz.std[5]) && abs(stz.mean[5] - 3) < 1.0

    # Errors: non-positive, non-divisor, and iterables
    @test_throws ArgumentError noisestats(datab; chans_per_band = 0)
    @test_throws ArgumentError noisestats(datab; chans_per_band = 6)
    @test_throws ArgumentError noisestats(datab; chans_per_band = 32)
    @test_throws ArgumentError noisestats([datab]; chans_per_band = 4)
end

using CUDA

if CUDA.functional()
    @testset "noisestats [CUDA]" begin
        gz = CuArray(zeros(Float32, 4, 4))
        @test noisestats(gz; robust = false) == (mean = 0.0f0, std = Inf)
        rngn = MersenneTwister(7)
        d2 = randexp(rngn, Float32, 64, 128)
        m, s = noisestats(CuArray(d2); robust = false)
        @test 0 < s < Inf
        # Per-channel stats through the GPU banded estimator
        gzb = noisestats(CuArray(zeros(Float32, 8, 4));
                         chans_per_band = 4, robust = false)
        @test gzb.mean == fill(0.0f0, 8) && gzb.std == fill(Inf, 8)

        # Banded non-robust statistics via the one-pass kernel:
        # matches the host path (which uses Statistics.std's
        # two-pass algorithm) to floating-point rounding
        rngk = MersenneTwister(21)
        gk = randexp(rngk, Float32, 64, 128)
        sg = noisestats(gk; chans_per_band = 8, robust = false)
        sh = noisestats(Array(gk); chans_per_band = 8, robust = false)
        @test sg.mean ≈ sh.mean rtol = 1e-4
        @test sg.std ≈ sh.std rtol = 1e-4
        # degenerate (all-zero) bands yield Inf sigma, like the host
        sz = noisestats(CuArray(zeros(Float32, 16, 8));
                        chans_per_band = 8, robust = false)
        @test sz.mean == fill(0.0f0, 16) && sz.std == fill(Inf, 16)
    end

    @testset "noisefloor [CUDA]" begin
        rngg = MersenneTwister(11)
        gm2 = randexp(rngg, Float32, 256, 256) .+
              randexp(rngg, Float32, 256, 256)
        gd = CuArray(gm2)
        @test noisefloor(gd; k = 2).mean ≈ noisefloor(gm2; k = 2).mean
        @test noisefloor(gd; k = 2).std ≈ noisefloor(gm2; k = 2).std
        @test noisestats(gd; robust = true, k = 2) ==
              noisestats(gm2; robust = true, k = 2)
        @test noisestats(CuArray(zeros(Float32, 4, 4)); robust = true) ==
              (mean = 0.0, std = Inf)
    end

    @testset "banded noisefloor [CUDA]" begin
        rngb = MersenneTwister(31)
        gb = randexp(rngb, Float32, 64, 128)
        gbd = CuArray(gb)
        # The batched device estimation matches the per-band host
        # fallback: the quantile selection is bit-exact (integer
        # histograms, host interpolation), so the unrefined (default)
        # mean and sigma match exactly
        sgb = noisestats(gbd; chans_per_band = 8)
        shb = noisestats(gb; chans_per_band = 8)
        @test sgb.mean == shb.mean
        @test sgb.std == shb.std
        # The opt-in clipped-mean refinement matches to reduction order
        sgr = noisestats(gbd; chans_per_band = 8, clip = 4)
        shr = noisestats(gb; chans_per_band = 8, clip = 4)
        @test sgr.mean ≈ shr.mean rtol = 1e-6
        @test sgr.std == shr.std
        # Bitwise reproducible across calls (deterministic partials)
        @test noisestats(gbd; chans_per_band = 8) == sgb
        # Keyword pass-through, with noisefloor's validation
        @test noisestats(gbd; chans_per_band = 8, qlo = 0.2).std ==
              noisestats(gb; chans_per_band = 8, qlo = 0.2).std
        @test noisestats(gbd; chans_per_band = 8, qhi = 0.4).std ==
              noisestats(gb; chans_per_band = 8, qhi = 0.4).std
        @test noisestats(gbd; chans_per_band = 8, k = 2) == sgb
        @test_throws ArgumentError noisestats(gbd;
                                              chans_per_band = 8,
                                              qlo = 0.7)
        @test_throws ArgumentError noisestats(gbd;
                                              chans_per_band = 8,
                                              qhi = 1.5)
        @test_throws ArgumentError noisestats(gbd;
                                              chans_per_band = 8,
                                              clip = 0.5)
        # Mixed degenerate (all-zero) band: band mean of zeros and
        # Inf sigma, like the host; other bands unaffected
        gz = CuArray(vcat(zeros(Float32, 16, 128), gb))
        sgz = noisestats(gz; chans_per_band = 16)
        hgz = noisestats(Array(gz); chans_per_band = 16)
        @test sgz.mean[1:16] == hgz.mean[1:16] == zeros(16)
        @test sgz.std[1:16] == hgz.std[1:16] == fill(Inf, 16)
        @test sgz.mean[17:80] ≈ hgz.mean[17:80] rtol = 1e-12
        @test sgz.std[17:80] == hgz.std[17:80]
        # Single-channel bands
        s1 = noisestats(gbd; chans_per_band = 1)
        h1 = noisestats(gb; chans_per_band = 1)
        @test s1.std == h1.std
        @test s1.mean ≈ h1.mean rtol = 1e-12
        # NaNs are rejected on the first pass, like fast_quantile
        gn = CuArray([1.0f0 NaN32; 3.0f0 4.0f0])
        @test_throws ArgumentError noisestats(gn; chans_per_band = 1)
    end
else
    @info "Skipping CUDA tests: no functional GPU available"
end
