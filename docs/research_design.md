# Research design

## Publication objective

Document the long-run relationship between Portugal's statutory minimum wage,
consumer prices and labour productivity from the introduction of the wage floor
in 1974, on a basis reconstructed from primary law; and establish what the
available regional variation can and cannot identify about the price
pass-through of minimum-wage policy.

This is a revision of the project's original objective, which was to estimate
pass-through causally from regional and sectoral exposure. That objective is not
attainable with the variation Portugal supplies, and the reasons are recorded
below rather than treated as unfinished work. Stating them is more useful to a
reader than a weakly identified estimate.

## Contribution architecture

### Contribution 1: the statutory series, from primary law

A reproducible statutory minimum-wage series covering 1974 to the present, with
every value read from the act that set it, retrieved from the Diário da
República by permanent identifier. It carries features that secondary
compilations flatten:

- three legally distinct minimum wages coexisted until 1991 and two until 2004,
  so a single pre-2005 series is a choice that must be stated;
- the official published history omits the act effective in 2000, while its 2001
  entries state their increase relative to it;
- the autonomous regions set their wages by different mechanisms, which matters
  for identification and not only for description;
- the statutory wage is paid fourteen times a year, so international
  compilations expressed on a twelve-month basis are about seventeen per cent
  higher than the figure in Portuguese law.

### Contribution 2: the long-run accounting

An annual series from 1974 tracking the minimum wage, CPI inflation, real labour
productivity, the real minimum wage, the wage floor relative to productivity, a
productivity-plus-lagged-inflation benchmark, and the residual of statutory
growth against that benchmark. The benchmark compounds rather than adds, which
matters at the inflation rates of the early 1980s.

This layer is descriptive and is the paper's principal empirical content. It is
not causal and is not presented as such.

### Contribution 3: a documented negative identification result

An account of why regional variation cannot identify pass-through in Portugal,
specific enough to be checked:

- the Azorean supplement is proportional, so in logs the Azorean statutory
  change equals the mainland's in every month after it took effect, and the
  region contributes no independent timing;
- Madeira is the only source of genuine divergence;
- with the small number of Portuguese regions, conventional cluster-robust
  inference rejects a true null far above its nominal size, and the estimates it
  calls highly significant survive none of a wild cluster bootstrap;
- a gap in the statutory register is not neutral, because an unregistered act
  appears as a frozen wage against a rising national one and enters the
  estimation as a shock that never happened.

## Estimands retained for the descriptive layer

For annual minimum wage growth `g_MW`, productivity growth `g_A` and inflation
`pi`:

```text
benchmark_t = (1 + g_A_t) * (1 + pi_{t-1}) - 1
residual_t  = g_MW_t - benchmark_t
```

The residual is an accounting object. A positive value means the wage floor rose
faster than the benchmark; it does not mean the increase caused inflation, and
the correlation between the residual and subsequent inflation cannot settle the
question, because statutory changes respond to the same conditions that drive
prices.

## Minimum publication standard

The original standard required pre-trend diagnostics, small-cluster inference,
predetermined exposure and robustness to alternative exposure definitions before
any causal conclusion. That standard is retained and is not met, which is why no
causal conclusion is drawn. The revised standard for what the paper does claim
is narrower and is met:

1. every statutory value is traceable to the act that set it;
2. every price and productivity series is retrieved reproducibly and checksummed;
3. the accounting identities are verified on every build;
4. no number in the manuscript is transcribed by hand;
5. the limits of identification are quantified rather than asserted.

## What the pass-through machinery is retained for

The estimators, few-cluster inference and falsification checks remain in the
repository and are exercised against the real panels. They support Contribution
3: the regional estimates are reported precisely to show the gap between what
conventional inference claims and what the bootstrap supports. They are not used
to make a causal claim.

## Identification threats

### Endogenous wage setting

Minimum-wage changes may respond to past inflation and expected economic conditions.

Mitigation:

- explicitly model lagged inflation and productivity;
- use predetermined exposure rather than contemporaneous bite;
- include calendar-time fixed effects in panel specifications;
- distinguish announced dates from implementation dates where possible.

### Regional confounding

High-bite regions differ structurally in tourism, sector mix, income, and unemployment.

Mitigation:

- region/category fixed effects;
- event-study pre-trend tests;
- region-specific trends as robustness only;
- tourism and unemployment controls;
- leave-one-region-out analysis;
- alternative baseline exposure years.

### Few regional clusters

Portugal has a small number of NUTS II regions.

Mitigation:

- do not rely on ordinary cluster-robust asymptotics alone;
- implement wild-cluster bootstrap;
- use randomization/permutation inference where defensible;
- exploit region × category observations but cluster at the policy-assignment level.

### Energy and tax shocks

The 2021-2023 inflation episode is dominated by shocks that are not minimum-wage policy.

Mitigation:

- HICP at constant tax rates;
- energy and food controls;
- imported inflation / import deflator;
- exclusions and separate estimates for energy-sensitive categories;
- explicit pandemic and Ukraine-war break indicators as robustness, not as substitutes for time fixed effects.

### Exposure measurement error

Observed minimum-wage coverage after a hike is endogenous to the policy.

Mitigation:

- freeze exposure before each policy episode;
- construct multiple bite definitions;
- use leave-one-year-out baselines;
- report attenuation sensitivity.

## Falsification tests

- Leads of the statutory shock must not predict pre-treatment inflation.
- Low-exposure CPI categories should show materially smaller responses.
- A fake implementation month should not generate the same event profile.
- Future minimum-wage increases should not predict current prices after controls.
- Results should not be driven solely by hospitality or one region.

## Minimum publication standard

Do not write a causal conclusion unless all of the following hold:

1. the data provenance is reproducible;
2. pre-trend diagnostics are acceptable;
3. small-cluster inference is implemented;
4. exposure is predetermined;
5. tax and energy shocks are addressed;
6. the result survives alternative exposure definitions;
7. the long-run macro layer and panel layer tell a coherent story without forcing agreement.

## Feasibility of the region-by-industry exposure measure

The exposure design requires a predetermined bite $b_{rs,0}$ that varies over
both regions and industries. The Portuguese sources that survive do not supply
one, and the gap is structural rather than a matter of effort.

**What exists.** The GEP monitoring reports give the share of full-time
employees paid the minimum wage **by economic activity, for the country as a
whole**. The variation across industries is large and real: in October 2017 it
ranged from 0.6 per cent in electricity and gas to 35.1 per cent in
accommodation and food, against a national average of 21.6 per cent. The
Quadros de Pessoal series gives employment **by economic activity** and,
separately, **by district**.

**What does not exist.** None of the 33 tables in the Quadros de Pessoal series
crosses a regional dimension with an industry dimension. Every regional table is
a marginal by district with no industry detail, and every industry table is
national.

**Why the marginals cannot substitute.** The shift-share alternative aggregates
national industry bites with regional industry weights,
$B_r = \sum_s w_{rs} b_s$. Recovering $w_{rs}$ from separate regional and
industry totals requires assuming that region and industry are independent,
which sets $w_{rs} = w_s$ and gives every region the national industry mix. The
resulting exposure is then *numerically identical in every region*, to
floating-point exactness. It would populate the column, pass every range check,
and identify nothing: the regressor would be absorbed by the fixed effects or
carry a coefficient estimated from no variation at all.

`construct_regional_bite` implements the shift-share aggregation for the case
where a genuine cross-tabulation is available, and `require_regional_variation`
refuses an exposure whose between-region variance share is zero. The test suite
demonstrates both branches, including that the marginal-derived construction
collapses to a single value.

**Even with a cross-tabulation, the assumption is substantive.** Holding the
bite constant within an industry across regions attenuates precisely the
variation the design exploits, because accommodation and food --- the highest-bite
industry by a wide margin --- differs across Portuguese regions in both its size
and its wage distribution. Anything built this way belongs in the robustness
section as an explicitly labelled variant, never in the baseline.

**Consequences for identification.** Portuguese statutory changes are national
on the mainland. If exposure carries no regional variation, the only remaining
variation is across consumption categories interacting with a common national
shock. That is a coherent design, and it is a different one: with calendar-time
fixed effects it identifies *relative* pass-through across categories rather
than its level, and inference can no longer rest on regional clusters, since the
effective number of independent shocks is the number of statutory changes. It is
not a substitute for the regional design and is not reported as one.

**What would unblock the specified design.** Any of: minimum-wage coverage
published by NUTS II region and economic activity jointly; Quadros de Pessoal
microdata, from which the cross-tabulation can be built directly; or the Madeira
and Azores statutory schedules, which would restore genuine regional variation
in the policy change itself and make a national bite far less damaging.


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
