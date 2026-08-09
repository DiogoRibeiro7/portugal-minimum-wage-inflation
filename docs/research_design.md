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
