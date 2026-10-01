# Thresholding statistics

For thresholding matrices of power-like data (e.g. frequency-by-drift-rate
or frequency-by-time), [`noisestats`](@ref) provides the (mean, sigma)
pair — robust by default — and [`noisenormalize!`](@ref) /
[`noisedenormalize`](@ref) convert between raw values and signal-to-noise
units.  The plain (`robust = false`) mode returns the ensemble statistics
of all elements (pooled over each band's elements in banded mode); the
robust mode is the recommended choice for data with any excess power
contamination.  The `chans_per_band` keyword estimates the statistics per
*band* of channels, returning per-channel values: the robust choice
whenever the noise power varies along the frequency axis (e.g. passband
power-level variations across a coarse channel that survive the analytic
filter-response correction), with the band width chosen small enough that
the variation within a band is negligible and large enough to pool many
samples.  On CUDA the robust per-band estimates are computed in a fixed
handful of batched device passes, at a cost independent of the number of
bands.

```@docs
noisestats
noisenormalize
noisenormalize!
noisedenormalize
```
