# Accuracy and tuning

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

## Choosing the quantiles

The `qlo` keyword (lower quantile paired with `qhi`) trades
contamination robustness against clean-data efficiency: lower quantiles are
less affected by excess power contamination (the contaminated samples sit in
the upper tail, and the quantile-value shift under contamination is
smallest where the Gamma density is steepest), while higher quantiles give
a more efficient spread estimate on clean data (the quantile correlation
with `qhi` grows with the quantile index).  Monte Carlo at `k = 51`
gives a clean-data split error of 0.68/0.61/0.37 pp for `qlo` of
0.05/0.1/0.2, versus a split bias of +4.3/+5.3/+6.8 pp at a 10%
contamination fraction; the `mean` estimate is largely `qlo`-independent
since `qhi` anchors it.  The default `qlo = 0.1` is a good compromise;
since contamination levels can vary widely, `qlo` can be
tuned per search.

The `qhi` keyword (upper quantile anchoring the mean and the upper end of
the spread estimate's span) spans a similar tradeoff: lower values sit
farther from the contamination and are biased less by it, while higher
values widen the quantile span and estimate the spread more efficiently.
Monte Carlo at `k = 51` on two-component noise with a 70/30 power split
(`N = 2^20`, 100 trials, `qlo = 0.1`; anchored estimates, i.e. the
default; reproducible with
`benchmarks/choosing_quantiles.jl`), with a contamination model of 10%
of samples replaced by powers drawn uniformly from 1x to 10x the floor:

| `qhi` | mean RMS% | std RMS% | split RMS (pp) | mean bias% | std bias% | split bias (pp) |
|-------|-----------|----------|----------------|------------|-----------|-----------------|
| 0.3   | 0.07      | 0.78     | 1.06           | +1.2       | +4.5      | +4.3            |
| 0.4   | 0.05      | 0.66     | 0.90           | +1.3       | +5.5      | +5.3            |
| 0.5   | 0.04      | 0.57     | 0.79           | +1.5       | +6.7      | +6.5            |
| 0.6   | 0.03      | 0.48     | 0.67           | +1.7       | +8.3      | +8.1            |
| 0.7   | 0.02      | 0.38     | 0.53           | +2.1       | +11.0     | +10.5           |

On clean data the higher quantiles are more efficient (the split error
roughly halves from `qhi = 0.3` to `0.7`); under contamination every
anchored estimate is biased less with the lower quantiles.  The default
`qhi = 0.5` (the median) is the midpoint of that tradeoff, and keeps the
mean's contamination response largely independent of `qlo`.

Note that this contamination model is *widespread and mild*, which the
opt-in clipped-mean refinement (enabled by `clip > 0`) handles less
gracefully
than rare strong outliers: contaminants below the `clip` threshold leak
into the survivor mean, adding a `qhi`-independent mean bias of about
+6% here (which also propagates into the split).  The optimal `clip`
depends on how the contamination is distributed in power, so there is no
general rule for choosing it.  Measured under the model above (`qhi = 0.5`,
same trials as the table above):

| `clip` | 1.0  | 1.5  | 2.0  | 3.0  | 4.0 (default) | 6.0   | 8.0   |
|--------|------|------|------|------|---------------|-------|-------|
| mean bias% | +1.4 | +0.2 | +0.6 | +2.6 | +6.3 | +23.3 | +45.0 |

(the plain mean bias is +45%).  Rare strong outliers are excluded by any
modest `clip` (the default 4 is fine); widespread contaminants well above
the floor are excluded by *lowering* `clip` toward the floor (raising
`clip` approaches the plain mean and is the worst choice under
contamination); contaminants overlapping the clean bulk cannot be
separated by any threshold.  A low `clip` also extrapolates more of the
distribution from the fitted shape, making the correction somewhat more
sensitive to shape error (the uptick at `clip = 1`).  When contamination
is suspected but uncharacterized, keep the refinement off (the default,
`clip = 0`):
the anchored estimates are the only choice whose robustness does not
depend on the distribution of the contamination.
