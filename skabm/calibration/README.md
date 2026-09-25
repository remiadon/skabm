# skabm.calibration

Two senses of "calibration", kept apart:

| | fits | against | module |
|---|---|---|---|
| **Population** | agent *rows* | accounting identities and known margins (Poledna §4; Deville & Särndal 1992) | `population` |
| **Parameters** | *behavioural parameters* θ | macro series (Fagiolo et al. 2019) | `parameters` |

The population is calibrated once, outside the search loop, and held frozen while θ moves.
The two compose the scikit-learn way: a population is a transformer on `X`, and a
parameter search is a search over `get_params()`.

## Populations

`make_dataset` draws columns from marginal samplers (`weighted_enum`, any polars-random
expression). A marginal sampler cannot impose joint structure: in the root README's
quickstart, `size` and `industry` are drawn independently, but Eurostat reports persons
per enterprise per industry. `GeneticConstraintCalibration` (and a Metropolis–Hastings
variant) permutes rows until per-industry mean size matches. Permutation leaves each
column's multiset untouched, so both marginals survive exactly, which is also why only
*relative* sizes can be imposed.

Calibrators follow the scikit-learn estimator API: `fit(frame)`, then `samplers_`.

## Parameters

skabm does not implement the search:
[black-it](https://github.com/bancaditalia/black-it) does it well, and
`simulator_model(world, free, ...)` is the adapter. It turns an `RDFSimulator`
configuration into the `model(theta, N, seed) -> (N, D)` callable black-it (and most
calibration toolboxes) expect. By default the `D` columns are the observables the rules
imply, so a method-of-moments loss needs no summary function. `stop=` ends a diverging
candidate early.

`noise_floor` is the diagnostic that belongs next to it. A simulated loss is an estimate,
so a loss gap between two candidates means nothing until it exceeds the spread one
candidate shows across seeds. Run it at the parameters you believe and at a clearly wrong
candidate: if the two ranges overlap, the observables do not identify that parameter.

`pip install "skabm[model-calibration]"` brings in black-it. Chapter 4 of
[notebooks/poledna](../../notebooks/poledna/poledna.ipynb) shows both in use.
