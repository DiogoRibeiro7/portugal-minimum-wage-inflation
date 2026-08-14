# Decision log

How positions in this project were reached, including the ones that were later
shown to be wrong. This is deliberately a record rather than a specification:
`docs/research_design.md` states what the design *is*, and anything here that
conflicts with it has been superseded.

Keeping the wrong turns is the point. Two of them were instructive, and both were
caught by building the thing that was said to be impossible.

## The region-by-industry exposure measure, revisited

An earlier version of this document concluded that the exposure design was
blocked because no source published minimum-wage coverage by region and
industry jointly. **That conclusion was wrong**, and it is recorded here rather
than quietly removed, because the way it was reached is instructive.

It was drawn from the Portuguese labour-ministry publications alone. None of the
33 tables in the recovered Quadros de Pessoal series crosses a regional with an
industry dimension, which is true, and it was generalised to "no source does",
which is false. Eurostat's regional accounts publish employment by NUTS II
region and NACE activity for Portugal from 2000 (`nama_10r_3empers`).

**What is and is not observed.** The shift-share measure

```text
B_r = sum_s q_rs,0 * b_s,0
```

needs regional industry composition `q_rs,0` and an industry bite `b_s,0`. The
composition is observed. The bite is observed nationally, from the GEP
monitoring reports, and is *not* observed regionally. Holding it constant within
industry is the measure's maintained assumption, and it is substantive:
accommodation and food service carries by far the highest bite, and Portuguese
regions differ in both how much of it they have and what it pays. The assumption
attenuates exactly the variation the design exploits.

So the design is constrained by an assumption, not blocked by missing data.
Those are different positions and the earlier document asserted the wrong one.

**What the measure delivers.** Frozen on 2015 composition, exposure ranges from
19.7 per cent in Grande Lisboa to 21.7 per cent in the Norte, with every one of
the nine regions taking a distinct value. The composition behind it varies
sharply: the Algarve has 41.8 per cent of employees in trade, transport,
accommodation and food service against the Alentejo's 21.6, while the Alentejo
has 13.4 per cent in agriculture against Grande Lisboa's 0.5.

**The aggregation must be employment-weighted, and it was not.** The regional
accounts publish coarser activity groups than the survey measures the bite at,
so each group's bite is the mean of its sections. Taking that mean unweighted
puts manufacturing and a near-zero-bite utility sector on equal footing: it
gives industry a bite of 13.7 per cent where the employment-weighted figure is
23.5. That is not a rounding difference. It reversed the regional ordering, with
the unweighted version placing the tourism-heavy Algarve top and the
manufacturing-heavy Norte bottom, which the weighted version exactly inverts.
National employment by NACE section supplies the weights, counting employees
rather than total employment, since the bite is a share of employees and
agriculture and construction carry far more self-employment than finance.

**Coverage must be measured, not assumed.** The survey excludes agriculture and
public administration. Agriculture is its own group and is straightforwardly
absent, but public administration shares a group with education and health, so
assigning the surveyed sections' bite to the whole group imputes a minimum wage
to workers nobody surveyed and reports the group as fully covered. Measuring it
properly, that group is 68.6 per cent covered and arts-and-other-services 47.8
per cent, which takes regional coverage down to a 73.4 to 89.4 per cent range
from the 78 to 99 previously reported.

**Predetermination is enforced, not assumed.** The published bite is dated
October 2017. Applying it to the 2015 policy restart would make exposure
post-treatment, since coverage after a rise is partly caused by it, and nothing
in the data prevents that. `check_predetermined` refuses a bite or composition
dated at or after the first shock. It currently refuses the 2015 episode, which
is correct: estimating that episode needs an earlier GEP snapshot, not a
different argument.

**Whether the variation is strong enough is a separate question.** The spread is
1.94 percentage points and the coefficient of variation 0.027. Nine distinct
values establish that a regional effect is identified in principle;
they do not establish that it is identified precisely enough to be informative,
and `require_regional_variation` answers only the first question.

`require_regional_variation` accepts this measure, having rejected the
marginals-derived construction it replaces. The guard was right about the old
input and says nothing about whether the assumption above is sound; that is a
judgement for the reader, which is why it is stated here rather than buried.

## Regional policy variation, recovered

The feasibility assessment above concluded that the exposure design was blocked
because no source crossed a region with an industry. That remains true of the
*bite*. It is no longer true of the *policy*.

Both autonomous regions legislate their own statutory minimum wage, and the
acts are retrievable from the Diário da República at stable permalinks. The
regional schedules are now built from those acts:

- Madeira legislates an explicit value. Its premium over the mainland varies —
  3.3 per cent in 2023 rising to 6.5 per cent in 2026 — and moves for reasons of
  regional politics that are not a function of the national figure.
- The Azores legislate a permanent 5 per cent supplement rather than an annual
  value.

This changes what the design can do. With regional variation in the policy
itself, the exposure term $\Delta \log MW_{rt}$ is no longer constant across
regions, so a national industry bite is far less damaging: identification can
come from regions facing genuinely different statutory changes at the same
moment, rather than only from cross-industry differences in a common shock.

Regional prices are now acquired. Statistics Portugal indicator `0014659`
publishes the consumer price index by NUTS II region and consumption purpose,
monthly from 1991, and the panel is built for 2000 onwards: nine regions,
fourteen consumption purposes. Both regions with their own statutory wage have
a complete monthly series, so prices and policy align on the same geographies.

What remains is not a data gap but a design constraint, and it should not be
described as though acquiring more data would resolve it. The two autonomous
regions are small, remote and tourism-intensive, so they are not a clean
control for the mainland: any estimate comparing them to it is exposed to
tourism cycles, transport costs and island-specific supply shocks that move
prices for reasons unrelated to wage policy. With two treated regions,
few-cluster inference is not a refinement but the whole problem, and the
procedures in `analysis/inference.py` are the minimum standard rather than a
robustness check. The falsification battery — pre-trends, placebo dating,
leave-one-region-out — carries correspondingly more weight than it would in a
design with many clusters.

## The region-by-category exposure, and a claim I got wrong

I twice advised against building the fuller structural exposure

```text
B[r, c] = sum_s q[r, s] * labour_cost_share[s] * consumption_bridge[c, s]
```

on the grounds that both new terms are national, so they would vary the shock
across consumption categories without widening the regional variation that both
estimated designs identify as the binding constraint. The README said the same.

That reasoning was wrong, and the error is worth recording because it is a
reasoning error rather than a data problem.

**What survives the fixed effects.** The design is `X[r,c,t] = B[r,c] * g[t]`
with `g[t]` the common national change, absorbed by `alpha[r,c]`, `lambda[r,t]`
and `mu[c,t]`. Because `g[t]` is common, the region-time effects remove the row
means of `B` and the category-time effects remove its column means. What
identifies the coefficient is therefore the double-demeaned, non-additive part of
the region-by-category matrix. Writing `B = Q diag(l) W'`, that matrix has rank
up to the number of sectors and is not additively separable, so the interaction
does not vanish. My claim that the national terms add nothing confused "these
factors do not vary across regions" with "their product does not vary across
region-category cells". Those are different statements.

**How much survives, on the real composition.** Taking the observed regional
employment shares and sweeping plausible bridges:

| Bridge | Variance surviving | Identifying spread |
| --- | --- | --- |
| Concentrated | 26 per cent | 6.8 points |
| Moderately concentrated | 19 per cent | 2.2 points |
| Diffuse | 7 per cent | 0.2 points |

against 4.1 points for the region-only measure the paper currently estimates.
So the answer depends entirely on how concentrated the consumption-to-industry
bridge is. A diffuse bridge is hopeless; a concentrated one gives *more*
identifying spread than the current design, over 117 region-category cells
rather than 9 regions.

Real COICOP categories are closer to the concentrated end than the diffuse one:
food draws on agriculture, food manufacturing and retail; restaurants on
accommodation and food service; transport on transport. That is an argument for
building the bridge, not against.

**And it would fix the seasonality confound.** The category-differential design
failed because it cannot carry calendar-time effects, so the January sales cycle
was read as a response to the January statutory rise. This design can carry
`mu[c,t]`, which absorbs each category's national January movement. What is left
is the variation of `B[r,c]` across regions within a category-month, which the
seasonal argument does not touch.

The sweep above is reproducible: run `docs/exposure_separability.py` from the
repository root against the built regional employment panel.

**What is still unresolved.** The figures above use simulated bridges and
simulated labour-cost shares; the real ones may be more diffuse than any of the
three. Building the bridge means supply-use tables and a COICOP-to-CPA
concordance, and the concordance involves judgement that should be recorded
rather than buried. The honest position is that the design is worth attempting
and was previously dismissed for a bad reason.

## Open work, as of August 2026

Two items remain from the external review. Both are specified rather than vague,
and neither depends on anything in a conversation.

### Make the inverted interval usable, then report it

`invert_bootstrap_interval` in `analysis/inference.py` is correct and tested: it
builds a confidence interval by inverting the restricted bootstrap test, so the
interval and the reported p-value agree by construction. An earlier attempt that
resampled around the estimate instead was discarded because it declared horizons
significant that the null-imposed test does not reject.

It is not wired into the pipeline because it is too slow. Each candidate value
calls the bootstrap afresh, and each call rebuilds the projection of a design
carrying one dummy per region-category and per month, so a seven-horizon path
costs hundreds of pseudo-inverses and does not finish in ten minutes.

The fix is already implemented once in the same file. `joint_wald_test`
precomputes `solve(X'X, X')` and reuses it across bootstrap draws, because only
the outcome changes; the same is true across inversion candidates. After that,
the interval belongs beside the p-value in `regional_design.tex` and
`exposure_design.tex`, and the manuscript should lead with it, because an
interval answers the question a null p-value cannot.

### Build the consumption bridge

`data/labour_shares.py` supplies one of the two terms the region-by-category
exposure needs, validated against Portugal's economy-wide labour share of 0.504.
The other is the production-to-consumption bridge: which industries supply each
COICOP category.

That needs supply-use tables and a COICOP-to-CPA concordance. The concordance
involves judgement, and this project's convention is that judgement is written
down rather than buried in a lookup table, so the mapping should be a documented
configuration file in the manner of `config/minimum_wage_bite.yaml` rather than a
dictionary in code.

Before building it, read the section above on the reasoning error, and run
`docs/exposure_separability.py`: the design is worth attempting only if the
bridge is concentrated, and the sweep says how much depends on that. A diffuse
bridge yields an identifying spread of a fifth of a percentage point and is not
worth the work.
