# Roadmap

What is left, and what is deliberately not. `docs/decision_log.md` records how
positions were reached and `docs/research_design.md` states what the design is;
this is the only forward-looking list, so that there are not two.

Current release: **v0.3.0** (14 August 2026),
[10.5281/zenodo.21939240](https://doi.org/10.5281/zenodo.21939240).

## Where v0.3.0 leaves the project

The causal layer is finished in the sense that matters: all three designs the
data admits are built and estimated, and they agree on why none of them
identifies a price response. The regional design has too few identifying events,
the exposure design cannot bound its coefficient, and the region-by-category
design bounds it away from every attainable value. The binding constraint is
that policy is assigned to nine regions and one supplies the divergence.

Nothing in the analysis is waiting on further code.

## Before 1.0.0

1.0.0 is a promise to freeze an interface, not a reward for volume, and v0.3.0
is where the interface moved most. Three things would make the promise
truthful.

- **Resolve `ptmw build policy-residual`.** It is a stub that prints a signpost
  to a function the caller must invoke themselves. Implement it or remove it; a
  placeholder in the public command surface is on its own enough to rule out a
  major version.
- **One release cycle without the estimator signatures moving.** v0.3.0 added
  `bootstrap_with_interval`, gave `estimate_panel_local_projections` three new
  arguments, added `build_absorbing_design`, and added fields to two result
  records. None of that is settled by assertion; it is settled by a cycle
  passing without further churn.
- **One release cycle without the reported numbers moving.** The repository's
  own rule makes an estimate change a breaking change, and v0.3.0 made one.

The manuscript being submitted or otherwise frozen is the natural moment for
all three to be true at once.

## Not blocked on this repository

Two data dependencies would change what the designs can do, and only their
publishers can lift them. Both are argued in `report/sections/robustness.tex`.

- Minimum-wage coverage published by NUTS II region **and** economic activity
  jointly. This removes the maintained assumption the exposure measure rests on
  and, more importantly, could widen a spread that industry mix alone cannot.
- A longer regional price series at consumption-purpose detail, which would
  allow pre-trend testing over a window long enough to be informative.

Neither relaxes the nine-cluster constraint.

## Known debt, worth doing or worth deciding against

None of these affect a published number. They are recorded so that the next
person does not rediscover them.

- **`construct_cost_exposure` and its family are exercised only by tests.**
  `apply_policy_shock`, `build_exposure_variants`, `exposure_correlation` and
  `construct_regional_bite` are in the same position. They want a bite that
  varies by region and industry, which is the measure nobody publishes. The
  docstring now says so. Either wire them to something or delete them with the
  reason recorded; leaving tested-but-unreachable code is the state that misleads.
- **`randomization_inference` is implemented, tested and unused by the
  pipeline.** It refits by least squares on every one of 9,999 permutations,
  which is unusable at the design sizes now in play. If it is to be reported
  alongside the bootstrap it needs the same treatment the bootstrap got.
- **`ptmw analyse structural-design` takes about twenty-two minutes** and
  dominates `make paper`. The cost is one pseudo-inverse of a 2,695-column
  design per horizon. Absorbing the fixed effects by within-transformation
  instead of explicit dummies would remove it, at the cost of the property the
  module currently guarantees: that the matrix bootstrapped is the matrix
  estimated. That trade has not been made and should not be made silently.
- **The download stages have not been re-exercised end to end recently.** The
  last full-pipeline verification ran every build and analysis stage plus the
  LaTeX build, but skipped `data download-sources` and the retrieval commands,
  which fail for upstream reasons rather than ours. A release that claims
  reproducibility from scratch should exercise them at least once.

## Deliberately not on the roadmap

**Adding more structure to the design.** This was tried in v0.3.0 and is the
most useful negative result the project has. The region-by-category design
carries region-time and category-time effects together, so it removes both the
island supply shocks that threaten the exposure design and the seasonal cycle
that defeats the category design — and it still cannot identify a price
response, returning magnitudes larger than complete pass-through of the entire
minimum-wage cost bill could produce. Structure adds cells, not clusters.

Anyone reaching for a fourth design should read the section on it in the
decision log first, and be able to say what their design does that this one did
not.
