# Research design

## Publication objective

Estimate whether increases in Portugal's statutory minimum wage generate measurable consumer-price inflation once productivity, pre-existing inflation, sectoral labour-cost exposure, and concurrent macroeconomic shocks are separated.

The paper must not identify the effect from a raw national time series alone. National minimum-wage changes are endogenous to inflation, productivity, political decisions, and the business cycle.

## Contribution architecture

### Contribution 1: long-run historical accounting

Build the first reproducible annual series, starting with the 1974 introduction, that jointly tracks:

- statutory minimum wage;
- CPI inflation;
- real labour productivity;
- real minimum wage;
- minimum wage relative to productivity;
- minimum-wage bite where available;
- a productivity-plus-lagged-inflation wage benchmark;
- the residual of actual minimum-wage growth relative to that benchmark.

This layer documents regimes and motivates the causal design. It is not by itself the core novelty claim.

### Contribution 2: regional and sectoral pass-through

Use public GEP data to measure the pre-existing share of workers covered by the RMMG across NUTS II regions and industries. Combine this with monthly regional CPI by consumption purpose from INE.

Where statutory schedules can be reconstructed, use Madeira and the Azores as additional policy variation because their regional minimum wage can differ from mainland Portugal.

### Contribution 3: structural price exposure

Construct a transparent cost-exposure index that maps industry-level minimum-wage exposure into consumption categories:

\[
E_{rct}=\sum_s b_{rs,0}\,\ell_{st}\,\omega_{cs}\,\Delta\log MW_{rt},
\]

where:

- \(b_{rs,0}\) is a predetermined minimum-wage bite for region \(r\) and sector \(s\);
- \(\ell_{st}\) is the labour-cost share;
- \(\omega_{cs}\) maps production industry \(s\) to consumption category \(c\);
- \(\Delta\log MW_{rt}\) is the applicable statutory minimum-wage change.

This is preferable to treating a 7% increase in the statutory minimum wage as a 7% economy-wide wage-cost shock.

## Primary estimand

For horizon \(h\), estimate the cumulative price response to a one-percentage-point increase in predicted minimum-wage-induced unit cost exposure:

\[
\Delta_h \log P_{r,c,t+h}
=\alpha_{rc}+\lambda_t+\beta_h E_{rct}+\Gamma X_{rct}+\varepsilon_{r,c,t+h}.
\]

The sequence \(\beta_h\) is the dynamic pass-through function.

## Alternative estimands

1. Price elasticity with respect to the statutory minimum wage, interacted with predetermined bite.
2. Inflation acceleration after a policy residual defined relative to productivity and lagged inflation.
3. Difference in price responses between labour-intensive and low-labour-intensity CPI categories.
4. Difference in pass-through between expansion, crisis, pandemic, and high-inflation regimes.

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
