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

### Read this before the rest

> **Every number below is a snapshot, not a source.** The manuscript takes its
> figures from generated macros; this document restates them by hand so the
> argument can be read on its own. Where the two disagree, the macros are right.
> Regenerate them with `ptmw build regional-exposure` and `ptmw analyse
> pass-through`, and read `report/tables/exposure_macros.tex` and
> `report/tables/identification_macros.tex` for the current values.

The document is layered: it records positions in the order they were reached,
including one that was later shown to be wrong, because how it was reached is
worth keeping. The current position is this, and it supersedes anything below
that conflicts with it.

- The exposure design is **not** blocked by missing data. Regional industry
  composition is published by Eurostat and the national minimum-wage bite by
  activity is recoverable from the labour ministry, so the shift-share measure
  is constructible and is constructed. The section *The region-by-industry
  exposure measure, revisited* corrects the earlier claim to the contrary, and
  the earlier claim is left in place, marked, rather than edited out.
- The exposure design **is** blocked by two other things. The measure spans 1.87
  percentage points across nine regions with a coefficient of variation of
  0.026, which identifies a regional effect in principle and nothing in
  practice; and the published bite post-dates the 2015 policy restart, so it is
  not predetermined for the episode with the most policy variation.
- Regional *policy* variation exists but is thin: three identifying
  region-months in one region, which conventional inference calls significant at
  six horizons and the bootstrap at none.

The net objective is unchanged, but the reason for it is different from the one
stated when it was written. The causal layer fails because the variation is too
small, not because the data does not exist. Those are different claims, and the
paper now makes the second.

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
specific enough to be checked. The claim is about the size of the variation, not
the availability of data, and every part of it is measured rather than asserted:

- the Azorean supplement is proportional, so in logs the Azorean statutory
  change equals the mainland's in every month after it took effect, and the
  region contributes no independent timing;
- Madeira is the only source of genuine divergence;
- the shift-share exposure measure that would substitute for policy variation is
  built and reported, and spans 1.87 percentage points across nine regions,
  which is too flat to identify a price response; the constraint is that the
  Portuguese wage floor binds at broadly similar rates across industries, so
  industry mix carries little information about exposure to it;
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

## Minimum publication standard, as revised

The original standard required pre-trend diagnostics, small-cluster inference,
predetermined exposure and robustness to alternative exposure definitions before
any causal conclusion. That standard is retained and is not met — small-cluster
inference is implemented and rejects nothing, and predetermination is enforced
and refuses the one episode worth estimating — which is why no causal conclusion
is drawn. The revised standard for what the paper does claim is narrower and is
met:

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
19.9 per cent in Grande Lisboa to 21.7 per cent in the Norte, with every one of
the nine regions taking a distinct value. The composition behind it varies
sharply: the Algarve has 38.8 per cent of employment in trade, transport,
accommodation and food service against the Alentejo's 20.5, while the Alentejo
has 22.0 per cent in agriculture against Grande Lisboa's 0.7.

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
per cent, which takes regional coverage down to a 66.6 to 88.4 per cent range
from the 78 to 99 previously reported.

**Predetermination is enforced, not assumed.** The published bite is dated
October 2017. Applying it to the 2015 policy restart would make exposure
post-treatment, since coverage after a rise is partly caused by it, and nothing
in the data prevents that. `check_predetermined` refuses a bite or composition
dated at or after the first shock. It currently refuses the 2015 episode, which
is correct: estimating that episode needs an earlier GEP snapshot, not a
different argument.

**Whether the variation is strong enough is a separate question.** The spread is
1.87 percentage points and the coefficient of variation 0.026. Nine distinct
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
