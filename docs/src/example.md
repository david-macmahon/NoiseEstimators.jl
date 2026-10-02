# Worked example: Voyager 2020 single coarse channel

The bandpass of that file is flat (coefficient of variation ~3%) across the
central 80% of its 2^20 channels and rolls off toward the band edges (the
-3 dB point is essentially at the edge), so the central 80% needs no
flattening.  Running `noisefloor` with `k = 51` (with the clipped-mean
refinement enabled, `clip = 4`; the default `clip = 0` shifts
the split by less than 0.1 pp at this sample size) on the central 80% of
the band (about 13.4 million samples, unflattened) recovers a
polarization split of 64.3%/35.7%, in agreement with the 62%/38% measured directly from
the full-polarization version of the same observation; the residual ~2 pp
is real-data systematics (residual bandpass and RFI) on top of the ~0.6 pp
statistical error at this sample size.
