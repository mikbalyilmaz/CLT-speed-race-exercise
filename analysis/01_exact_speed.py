"""Analysis 1: the exact speed of convergence, and what determines it.

Why: version 1 ranked the distributions by eye, from simulations. Here the Kolmogorov distance
D_n = sup_x |P(Z_n <= x) - Phi(x)| is computed exactly for the six distributions that allow it, and compared
with the leading term of its expansion: |skewness| phi(0) / (6 sqrt n) for skewed distributions,
|excess kurtosis| HE3_MAX / (24 n) for symmetric ones (Edgeworth expansion), and a separately derived
constant / sqrt(n) for Student-t(3), whose third absolute moment is infinite.

Design: 41 values of n from 1 to 10^4, evenly spaced on a log scale. For every distribution, also the smallest
n with D_n < 0.01, found by computing D_n for every n up to that point and checking the next 200 values.
"""

import math

import numpy as np

from common import EXACT, THRESHOLD, Distribution, name, save_table
from cltspeed import kolmogorov_distance, leading_term, v1_distributions
from cltspeed.montecarlo import noise_floor
from cltspeed.style import AQUA, BLUE, GOLD, MUTED, ORANGE, PINK, PURPLE, L, save_figure, use_style

NS = np.unique(np.round(np.geomspace(1, 10_000, 41)).astype(int))
COLOURS = [BLUE, AQUA, PURPLE, PINK, GOLD, ORANGE]
MARKERS = ["o", "s", "^", "v", "D", "P"]
FLOOR_R2000 = noise_floor(2000)[0]


def first_n_below(dist, level, check_after=200):
    """Smallest n with D_n < level such that D_m < level also for the next `check_after` values of m."""
    n = 1
    while True:
        if kolmogorov_distance(dist, n)[0] < level:
            if all(kolmogorov_distance(dist, m)[0] < level for m in range(n + 1, n + check_after + 1)):
                return n
        n += 1


rows, curves = [], {}
for dist in EXACT:
    ds = []
    for n in NS:
        d, x, method = kolmogorov_distance(dist, int(n))
        lead = leading_term(dist, int(n))
        rows.append({"distribution": dist.label, "n": int(n), "D_n": d, "x_at_max": x, "method": method,
                     "leading_term": lead[0], "ratio_to_leading_term": d / lead[0]})
        ds.append(d)
    curves[dist.label] = np.array(ds)
    print(f"{dist.label}: D_1 = {ds[0]:.4g}, D_10000 = {ds[-1]:.4g}")
save_table(rows, "table1_exact_distance")

# One row per distribution of version 1 (plus Student-t(3)): moments, leading term, exact distances.
summary = []
for dist in v1_distributions()[:8] + [Distribution("Student-t", {"df": 3})] + v1_distributions()[8:]:
    lead = leading_term(dist, 1) if dist.clt_applies() else None
    exact = dist.label in curves
    row = {"distribution": dist.label,
           "mean_exists": dist.mean() is not None, "variance_exists": dist.sd() is not None,
           "skewness": dist.skewness() if dist.clt_applies() else "", "excess_kurtosis":
           dist.excess_kurtosis() if dist.clt_applies() and dist.excess_kurtosis() is not None else "",
           "leading_order": lead[1] if lead else "", "leading_constant": lead[0] if lead else "",
           "leading_source": lead[2] if lead else ""}
    for n in (10, 30, 100, 1000, 10000):
        row[f"D_{n}"] = kolmogorov_distance(dist, n)[0] if exact else ""
    row["n_below_0.01_exact"] = first_n_below(dist, THRESHOLD) if exact else ""
    row["n_below_0.01_leading_term"] = (math.ceil((lead[0] / THRESHOLD) ** (1 / lead[1])) if lead else "")
    row["n_below_noise_R2000_exact"] = first_n_below(dist, FLOOR_R2000) if exact else ""
    if dist.name == "Normal":
        row.update({"D_10": 0.0, "D_30": 0.0, "D_100": 0.0, "D_1000": 0.0, "D_10000": 0.0,
                    "n_below_0.01_exact": 1, "n_below_noise_R2000_exact": 1})
    summary.append(row)
    print(row["distribution"], row["n_below_0.01_exact"], row["n_below_0.01_leading_term"])
save_table(summary, "table2_summary")

use_style()


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    ax = axes[0]
    for (label, ds), c, mk in zip(curves.items(), COLOURS, MARKERS):
        dist = next(d for d in EXACT if d.label == label)
        lead = np.array([leading_term(dist, int(n))[0] for n in NS])
        ax.loglog(NS, ds, color=c, lw=1.2, marker=mk, ms=2.6, markevery=4, label=name(label))
        ax.loglog(NS, lead, color=c, lw=0.8, ls="--")
    ax.axhline(THRESHOLD, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("Exact distance $D_n$", "Kesin uzaklık $D_n$"))
    ax.set_title(L("Exact distance (dashed: leading term)", "Kesin uzaklık (kesikli: öncü terim)"))
    ax.set_ylim(1e-6, 0.3)
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1]
    for (label, ds), c, mk in zip(curves.items(), COLOURS, MARKERS):
        dist = next(d for d in EXACT if d.label == label)
        lead = np.array([leading_term(dist, int(n))[0] for n in NS])
        ax.semilogx(NS, ds / lead, color=c, lw=1.2, marker=mk, ms=2.6, markevery=4, label=label)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 2.2)
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("$D_n$ / leading term", "$D_n$ / öncü terim"))
    ax.set_title(L("Ratio to the leading term", "Öncü terime oran"))
    from cltspeed.style import decimal_axis
    decimal_axis(ax)
    return fig


save_figure("fig1_exact_speed", draw,
            lambda: "Figure 1. Exact distance between the standardized mean and the normal distribution")
