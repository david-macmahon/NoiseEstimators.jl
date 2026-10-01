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

## Choosing `qlo`

The `qlo` keyword (lower quantile paired with the median) trades
contamination robustness against clean-data efficiency: lower quantiles are
less affected by excess power contamination (the contaminated samples sit in
the upper tail, and the quantile-value shift under contamination is
smallest where the Gamma density is steepest), while higher quantiles give
a more efficient spread estimate on clean data (the quantile correlation
with the median grows with the quantile index).  Monte Carlo at `k = 51`
gives a clean-data split error of 0.68/0.61/0.37 pp for `qlo` of
0.05/0.1/0.2, versus a split bias of +4.3/+5.3/+6.8 pp at a 10%
contamination fraction; the `mean` estimate is largely `qlo`-independent
since the median anchors it.  The default `qlo = 0.1` is a good compromise;
since contamination levels can vary widely, `qlo` can be
tuned per search.
