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
**That last figure is not comparable to the others and should not be used as the
benchmark**: it comes from the same simulation, with labour shares drawn from a
uniform and no bite at all. On the real terms the region-only measure gives 1.4
points. The section below, on the bridge as built, has the numbers that matter.
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

## The inverted interval, made usable and reported

This was the first of the two items left by the external review, and it is done.
The interval is now beside the p-value in `regional_design.tex` and
`exposure_design.tex`, and the manuscript leads with it. What the work turned up
along the way is worth more than the speed-up.

**The projection was the smaller half of the cost.** The plan recorded here was
to reuse `solve(X'X, X')` across candidates, as `joint_wald_test` already does.
That was right but incomplete: on the regional design the pseudo-inverse cost
about a second per candidate and the *second* least-squares solve — the fit of
the reduced design, which imposes the null — cost twelve. Both depend only on
the design, but the second does not need computing at all. Frisch-Waugh gives
the restricted fit from the full projection directly,

```text
restricted_fit = X b - b_t * x_perp
```

and the projector already holds `x_perp`: the target row of `(X'X)^-1 X'` is
`x_perp / (x_perp' x_perp)`. Twelve seconds became two milliseconds.

**The sign space is smaller than the number of draws.** The wild *cluster*
bootstrap gives each cluster's whole residual block one sign, so every resampled
outcome — all 512 of them — is a combination of the same ten vectors: the
restricted fit and the residuals masked to one cluster at a time. Projecting
those ten and recombining is not an approximation to the per-draw loop, it is
the same arithmetic with the shared work done once. A horizon's interval takes
under two seconds against about fifty minutes for the design; the ordinary
bootstrap p-value went from ten seconds to under a third of one.

**A tie was being settled by rounding, and it moved published numbers.** Making
the draws cheap made a latent defect visible. The enumerated sign space contains
the all-positive vector, which rebuilds the original sample exactly, so its
statistic *is* the observed statistic and must count — and because the
statistics are odd in the sign vector, so must its negation. Both were being
compared with a bare `>=` against a value computed by a different route, so
whether they counted depended on which way the last bits fell. On horizons where
whole clusters contribute no variation each distinct statistic repeats and the
effect was larger still: eight draws of 512 at one horizon. The symmetry is now
imposed rather than recomputed, the tie is settled by a measured tolerance, and
every bootstrap p-value in the paper moved by up to 0.016. No conclusion moved;
the before-and-after values are in the changelog.

**The search range was built from the statistic the module distrusts.** The
range was quoted in cluster-robust standard errors, and it failed in exactly the
direction that matters: where the clustered error most understates the
uncertainty, the bootstrap interval is widest and the range built from it is
narrowest. At impact on the regional design a t of 8.2 carries a bootstrap p of
0.23, so every candidate within six standard errors survived and the search
never reached zero. Reported as it stood, the table would have shown a tight
interval excluding zero beside a p-value of 0.23. The search now widens until
the interval closes, and says so when it cannot.

**What the intervals show.** The exposure design's widest runs from -73.7 to
64.7. It admits full pass-through, sixty times full pass-through, and the same
magnitudes negative. That is the point of reporting it: a table of nulls cannot
distinguish a design that found no effect from one that could not have found
any, and this one is emphatically the second.

## The consumption bridge, built, and what it turned out to be

The gate was that the design is worth attempting only if the real bridge is
concentrated. It is built, the answer is that it is moderately concentrated
rather than diffuse, and the gate passes. `config/consumption_bridge.yaml`
records the concordance, `data/supply_use.py` the measurement, and
`docs/consumption_bridge_feasibility.py` reproduces the numbers below.

**The gate, answered.** Against simulated bridges the sweep gave 6.77 points
when concentrated, 2.22 moderately so, and 0.23 when diffuse. The real bridge
gives **2.31**, and 2.00 to 2.89 across the three defensible ways of allocating
the trade margin. It sits at the moderate end and nowhere near the diffuse one,
so the design is worth estimating.

**The benchmark in the table above was wrong, and flattering to the wrong side.**
That table quotes 4.13 points for the region-only measure the paper estimates,
and it is not comparable: it came from the same simulation, which drew labour
shares from a uniform and left the bite out altogether. Recomputed on the real
bite and the real labour shares, the region-only measure gives **1.40** points.
So the fuller design does not deliver half of what the current one has, as the
old comparison implied; it delivers 1.4 to 1.7 times as much depending on the
coverage weighting, across 117 region-category cells rather than 9 regions. Two
numbers from different generating processes were being compared as though they
were the same quantity.

**What the measurement decided, more than the concordance did.** Two arguments
to the use-table request move the answer further than any judgement in the
concordance.

The table has to be read at *basic* prices. At purchasers' prices the retail
margin on a loaf of bread sits inside "food products", so the shelf price is
credited to the industry that baked it rather than the one that sold it. The two
conventions describe Portuguese consumption differently enough to change which
industry the shopping basket's wage exposure belongs to: food, beverages and
tobacco are 17.0 per cent of the basket under one and 7.9 under the other, while
wholesale and retail trade go from almost nothing to 18.6. Retail is
minimum-wage intensive and manufacturing much less so.

And it has to be read at *domestic* uses. A Portuguese wage rise does not raise
the cost of the imported content of a television. Textiles and apparel are 7.4
per cent of the basket at purchasers' prices including imports and 2.0 per cent
of domestic content at basic prices, because most of what Portugal wears is
imported and most of what its textile industry makes is exported.

**What no concordance can reach.** Domestic content at basic prices is 91.0 of
the 121.9 billion euro Portuguese households spent in 2015. The remaining
quarter is imported content (12.2 per cent) and taxes less subsidies on products
(13.2), and neither has a Portuguese producing industry behind it. That is a
ceiling on the whole exercise and it is a property of the economy rather than of
the data.

**The rule that works against the design, kept anyway.** Wholesale and retail
trade are nearly a fifth of domestic consumption at basic prices, and nobody
buys retail trade services as such: the margin is earned on the goods it
distributes. Spreading it across categories in proportion to their goods content
is what puts retail employment behind the basket, and it is also what makes the
bridge more diffuse, because it drops a large common block of one industry into
every goods-carrying category, which is exactly the additive structure the fixed
effects absorb. Leaving the margin unallocated would raise the identifying
spread from 2.31 to 2.89 and would be indefensible. The variants are named in
the config and measured in the feasibility script rather than chosen quietly.

**What the ten-sector ceiling costs.** The regional accounts publish ten
activity groups, and one of them carries trade, transport, accommodation and
food service together. No concordance can separate a retail margin from a
restaurant meal, so the two categories where a minimum wage should bite hardest
load on the same column of `Q`. Two categories come out identical: alcohol and
tobacco, and clothing and footwear, both being manufactured goods plus a
proportional margin, contribute one distinct row between them.

**What it does not fix.** Inference clusters on region because policy is
assigned by region, and going from 9 regions to 117 region-category cells adds
observations within clusters rather than clusters. The binding constraint the
rest of the paper documents — few clusters, one genuinely treated region — is
untouched by this design.

What it does buy is the fixed effects. The design carries `lambda[r,t]` and
`mu[c,t]`, so region-time shocks are absorbed: tourism cycles, transport costs
and island-specific supply shocks, which the manuscript names as the main threat
to comparing the autonomous regions with the mainland, no longer enter the
coefficient. The category-time effects absorb the January sales cycle that
defeated the category-differential design. That, rather than the wider spread,
is the reason to estimate it.

## The region-by-category design, estimated, and what disqualifies it

Built and estimated. `ptmw build structural-exposure` composes it and
`ptmw analyse structural-design` estimates it, absorbing region-category,
region-time and category-time effects together. It is the only design in the
paper that rejects anything, and it is disqualified by its own magnitudes.

**The result.** Two of seven horizons reject at five per cent by the bootstrap,
at p = 0.016 and 0.027. None survives the Holm correction across the horizon
family, whose smallest adjusted value is 0.109. The joint pre-trend test over
five leads does not reject, at p = 0.646, which is the most comfortable pass any
design in this paper achieves.

**What condemns it is the scale, not the significance — and the first version of
this entry got the scale wrong.** I wrote that the regressor is a cost share
times a log wage change, so complete pass-through is a coefficient of one and
13.9 is fourteen times it. That is false, and it was nearly published. The
exposure weights each industry by the region's *employment share* in it, which is
what supplies the regional dimension and is also a fraction summing to one across
industries, so `B` is a cost share scaled down by that weight. The ratio of the
two runs from 0.009 to 0.418 across cells, median 0.21, so a coefficient of one
is nowhere near complete pass-through and — the part that matters — there is no
fixed factor that would convert it, because the scaling differs cell by cell.

The check that does work needs no unit assumption, because it puts both sides in
points. At eighteen months the estimate implies a differential price response of
2.7 percentage points between the most and least exposed cells for a ten per cent
statutory rise. The most minimum-wage-intensive consumption category has 15 per
cent of its costs in minimum-wage labour, so if every euro of that reached prices
a ten per cent rise could move it by at most 1.5 points against a category with
none. Three of the seven horizons imply more than that ceiling, including both
significant ones. Those estimates are not large, they are unattainable. The
profile also alternates sign between adjacent horizons, which no cumulative
response function does.

`structural_exposure` now carries `category_cost_share` for exactly this reason:
the ceiling is a computed quantity rather than a remembered one, and the
exposure's own docstring says it is not a cost share so the next reader does not
repeat the error.

**The pre-trend pass is not a defence, and the reason generalises.** Passing a
lead test establishes that the leads carry no signal. It says nothing about
whether the contemporaneous coefficient is a magnitude the economics permits. A
design can be clean on timing and still be dividing by a number close to zero,
and the identifying spread of 1.92 points is that number. A falsification battery
that tests only timing will pass a design that is failing on scale.

**The identifying spread is 1.92 points, not the 2.31 the feasibility script
first gave.** The difference is the coverage weighting: `structural_exposure`
multiplies each activity's contribution by the share of its employment the bite
was actually measured on, and the script's own arithmetic did not. The
production measure is the conservative one and it is the one to quote; both are
far above the diffuse case that would have stopped the work.

The script now calls `structural_exposure` for its headline rather than
recomputing, so there is one number and one code path. That fix was itself
instructive: passing the bite as a bare column instead of the full frame silently
drops `measured_employment_share` and turns the coverage weighting off, which
reproduces 2.31 while looking like it is calling the production builder. A
function whose behaviour depends on whether an optional column survived a
`set_index` is a trap, and the comment at the call site now says so.

**What the exercise settles.** The paper's causal layer rests on the claim that
the binding constraint is the assignment of policy to nine regions, not any
missing structure. That claim was previously an argument. It is now a result: the
structure was built, it absorbs the island supply shocks the exposure design is
exposed to and the seasonal cycle that defeats the category design, and the
answer did not improve. Removing the confounds did not help because the confounds
were never the constraint. A design with more structure adds cells and not
clusters.

**A performance note worth keeping.** The three-way design carries 2,695 columns
against 12,077 rows, and estimating it exposed that each horizon was paying for
its decomposition four times over: once for the point estimate through a
least-squares solve that alone cost about ten minutes at this size, once for the
cluster-robust variance, once for the bootstrap and once for the interval.
`bootstrap_with_interval` pays for it once. That took the design from
uncomputable to twenty-two minutes, and it also closed a gap in the reporting:
the estimate printed beside a bootstrap p-value is now the estimate that p-value
was computed from by construction rather than by coincidence.

## Open work, as of August 2026

Nothing is specified and outstanding. The two items the external review left have
both been carried through, and the third that fell out of the second --- building
and estimating the region-by-category design --- is done.

What remains is not work on this repository but two data dependencies that only
their publishers can lift, both recorded in `report/sections/robustness.tex`:
minimum-wage coverage published by NUTS II region and economic activity jointly,
which would remove the exposure measure's maintained assumption and could widen a
spread that industry mix alone cannot; and a longer regional price series at
consumption-purpose detail, which would allow pre-trend testing over an
informative window.

Neither addresses the constraint all three designs now point at. Policy is
assigned to nine regions and one supplies the divergence, so inference clusters
on nine however finely the panel is cut. Anyone tempted to add structure to fix
that should read the section above first: it was tried, and the answer is that
structure adds cells rather than clusters.
