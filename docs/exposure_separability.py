"""How much of B_rc survives region-time and category-time fixed effects?

The proposed design is X_rct = B_rc * g_t with B_rc = sum_s q_rs * l_s * w_cs,
absorbed by alpha_rc, lambda_rt and mu_ct. Since g_t is common, lambda_rt removes
the row means of B and mu_ct removes the column means, so what identifies beta is
the double-demeaned (non-additive) part of the region-by-category matrix.

If B were additively separable in r and c, nothing would survive. B = Q diag(l) W'
has rank at most the number of sectors, so it is not separable in general --- but
"not zero" and "enough" are different claims, and only the second matters.
"""

import numpy as np
import pandas as pd

from pt_mw_inflation.data.eurostat_regional import industry_shares

shares = industry_shares(pd.read_parquet("data/processed/regional_employment.parquet"), year=2015)
Q = shares.pivot(index="region", columns="activity", values="employment_share").fillna(0.0)
sectors = list(Q.columns)
regions = list(Q.index)
print(f"Q: {len(regions)} regions x {len(sectors)} sectors")

rng = np.random.default_rng(20260814)
# Labour-cost shares: national constants, plausible spread across sectors.
labour = pd.Series(rng.uniform(0.20, 0.55, size=len(sectors)), index=sectors)

CATEGORIES = 13


def double_demean(matrix: np.ndarray) -> np.ndarray:
    """Remove row and column means: what alpha_rc, lambda_rt and mu_ct leave."""
    return (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )


def report(label: str, bridge: np.ndarray) -> None:
    """Share of exposure variation surviving the two-way absorption."""
    B = Q.to_numpy() @ np.diag(labour.to_numpy()) @ bridge
    total = float(np.var(B))
    surviving = float(np.var(double_demean(B)))
    spread = 100 * (B.max() - B.min())
    residual_spread = 100 * (double_demean(B).max() - double_demean(B).min())
    print(
        f"{label:<34} total sd {100 * np.sqrt(total):6.3f}pp | "
        f"surviving {100 * surviving / total:5.1f}% of variance | "
        f"raw spread {spread:5.2f}pp -> identifying spread {residual_spread:5.2f}pp"
    )


# A concentrated bridge: each consumption category drawn mostly from one sector.
concentrated = np.zeros((len(sectors), CATEGORIES))
for c in range(CATEGORIES):
    concentrated[c % len(sectors), c] = 0.8
    concentrated[:, c] += 0.2 / len(sectors)
report("concentrated bridge", concentrated)

# A diffuse bridge: every category draws from every sector in similar proportion.
diffuse = np.full((len(sectors), CATEGORIES), 1.0 / len(sectors))
diffuse += rng.uniform(-0.02, 0.02, size=diffuse.shape)
diffuse = np.clip(diffuse, 0.0, None)
diffuse /= diffuse.sum(axis=0, keepdims=True)
report("diffuse bridge", diffuse)

# A realistic middle: moderate concentration with a common services component.
middle = rng.dirichlet(np.full(len(sectors), 0.7), size=CATEGORIES).T
report("moderately concentrated bridge", middle)

# For comparison: the region-only measure the paper already estimates, which
# keeps its full cross-region spread because only lambda_t is absorbed.
B_r = Q.to_numpy() @ labour.to_numpy()
print(
    f"\n{'region-only B_r (current design)':<34} "
    f"spread {100 * (B_r.max() - B_r.min()):5.2f}pp across {len(regions)} regions"
)
