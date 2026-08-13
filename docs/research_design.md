# Research design

## Publication objective

Document the long-run relationship between Portugal's statutory minimum wage,
consumer prices and labour productivity from the introduction of the wage floor
in 1974, on a basis traceable to the acts that set it; and establish what the
available regional variation can and cannot identify about the price
pass-through of minimum-wage policy.

This is a revision of the project's original objective, which was to estimate
pass-through causally from regional and sectoral exposure. That objective is not
attainable with the variation Portugal supplies, and the reasons are recorded
here rather than treated as unfinished work. Stating them is more useful to a
reader than a weakly identified estimate.

### Read this before the rest

> **Every number below is a snapshot, not a source.** The manuscript takes its
> figures from generated macros; this document restates them by hand so the
> argument can be read on its own. Where the two disagree, the macros are right.
> Regenerate them with `ptmw build regional-exposure` and `ptmw analyse
> pass-through`, and read `report/tables/exposure_macros.tex` and
> `report/tables/identification_macros.tex` for the current values.
>
> The per-group figures below --- each group's bite, and how much of its
> employment the survey covered --- have no macro behind them, because the paper
> does not quote them. Reproduce those from
> `activity_bite_from_registry(registry, national_employment)` rather than
> looking for a generated file that does not exist.

This document states the current design. How these positions were reached,
including two that were later shown to be wrong, is in `docs/decision_log.md`.

- The exposure design is **not** blocked by missing data. Regional industry
  composition is published by Eurostat and the national minimum-wage bite by
  activity is recoverable from the labour ministry, so the shift-share measure
  is constructible and is constructed. The earlier claim to the contrary, and
  how it was reached, are in `docs/decision_log.md`.
- The exposure design is **estimated**, not merely described. The published
  table carries five survey rounds; the October 2015 round is predetermined for
  every shock from 2016 onwards. Estimated on it, with calendar-time fixed
  effects, not one of the seven horizons is significant at five per cent even by
  conventional clustered inference, whose smallest p-value is 0.48. The measure
  spans 2.49 percentage points across nine regions, coefficient of variation
  0.035. That is a stronger negative than the policy design gave: inference
  known to over-reject cannot manufacture even one false positive here.
- Regional *policy* variation exists but is thin: ten identifying region-months
  in one region. Conventional inference calls five horizons significant, the
  bootstrap leaves one, and Holm's correction across the seven-horizon family
  leaves none. The binding constraint is that there is one treated region, not
  that there are too few dates --- which is why recovering Madeira's missing
  acts improved the point estimates without changing the conclusion.

The net objective is unchanged, but the reason for it is different from the one
stated when it was written. The causal layer fails because the variation is too
small, not because the data does not exist. Those are different claims, and the
paper now makes the second.

## Contribution architecture

### Contribution 1: the statutory series, traceable to the acts

A reproducible statutory minimum-wage series covering 1974 to the present. The
national series is parsed from the labour ministry's register and reconciled
against both the increases that register states and Eurostat's independent
compilation; the acts the register omits, and both regional schedules, are
retrieved from the Diário da República by permanent identifier and read for the
values they set. Describing the whole series as "read from the acts themselves"
would overstate it: the national baseline is the register, verified. It carries features that secondary
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
- Madeira is the only source of genuine divergence, and only from 2017. Until
  2016 its legislated figures follow a two per cent rule on the mainland amount,
  stated in the preamble to DLR 18/2016/M and obeyed to the cent, so Madeira was
  on the same kind of rule as the Azores and contributed no independent timing
  either. This is checked arithmetically on the statutory panel rather than
  asserted, and emitted as a macro;
- the shift-share exposure measure that would substitute for policy variation is
  built and reported, and spans 2.49 percentage points across nine regions on
  the October 2015 round the design is estimated with, which is too flat to
  identify a price response; the constraint is that the
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
- Low-exposure CPI categories should show materially smaller responses. Note
  that this test is not currently available: 94 per cent of statutory log change
  falls in January, so the shock is nearly a January indicator, and the category
  ranking measures each category's own January seasonal rather than its response.
  Clothing falls 15.9 per cent in an average January on winter sales, which is
  why its coefficient looked broken; nothing is wrong with the series.
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

## Where the superseded reasoning went

Two conclusions in this project were reached, acted on, and later reversed: that
no source crossed a region with an industry, and that the exposure design was
therefore blocked. Both were wrong, and how they were reached is recorded in
`docs/decision_log.md` rather than here, so that this document describes only
the current design.
