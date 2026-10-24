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
from cltspeed.style import RED, SERIES, L, auto_axis, fit_log_axis, may_touch, save_figure, use_style

NS = np.unique(np.round(np.geomspace(1, 10_000, 41)).astype(int))
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
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.6), constrained_layout=True)
    ax = axes[0]
    ratios = []
    for (label, ds), (c, ls, mk) in zip(curves.items(), SERIES):
        dist = next(d for d in EXACT if d.label == label)
        lead = np.array([leading_term(dist, int(n))[0] for n in NS])
        ax.plot(NS, ds, color=c, ls=ls, lw=1.1, marker=mk, ms=2.2, markevery=4, label=name(label))
        ax.plot(NS, lead, color=c, lw=0.6, ls=(0, (3, 2)))
        ratios.append(np.asarray(ds) / lead)
    may_touch(ax.axhline(THRESHOLD, color=RED, lw=0.9, ls=(0, (1, 1.4))))
    fit_log_axis(ax, "x", NS, every=2)
    log_lo = 1e-6
    fit_log_axis(ax, "y", [log_lo, 1.0], every=2)
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_title(L("(a) Exact distance $D_n$", "(a) Kesin uzaklık $D_n$"))
    ax = axes[1]
    for (label, ds), (c, ls, mk), r in zip(curves.items(), SERIES, ratios):
        ax.plot(NS, r, color=c, ls=ls, lw=1.1, marker=mk, ms=2.2, markevery=4)
    may_touch(ax.axhline(1, color=RED, lw=0.9, ls=(0, (1, 1.4))))
    fit_log_axis(ax, "x", NS, every=2)
    auto_axis(ax, "y", np.append(np.concatenate(ratios), 0.0))
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_title(L("(b) $D_n$ / leading term", "(b) $D_n$ / öncü terim"))
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=6, handlelength=2.4, ncol=2, columnspacing=0.8)
    return fig


save_figure("fig1_exact_speed", draw,
            lambda: "Figure 1. Exact distance between the standardized mean and the normal distribution "
                    "(dashed: leading term; red dotted: 0.01 and 1)")
