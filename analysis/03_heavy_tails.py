"""Analysis 3: the distributions without an exact method, and the ones where the CLT does not apply.

Why: version 1 includes Lognormal, F and Pareto, for which no closed-form characteristic function is
available here, and it stops with a message for Cauchy and for Pareto with tail index <= 2, where the
variance is infinite. Two questions remain: how fast do the first three converge, and what does the sample
mean do when the variance is infinite?

Design: independent paths up to n = 10^4 (25 sample sizes, evenly spaced on a log scale): R = 100,000 paths
for Lognormal(0.6), F(5, 10) and Pareto(2.5), R = 20,000 for Pareto(1.5) and Cauchy.
(a) Interquartile range (IQR) of the sample mean for Pareto(2.5), Pareto(1.5) and Cauchy; the IQR exists
    even when the variance does not. Normal(0, 1) is shown exactly: IQR = 2 * 0.6745 / sqrt(n).
(b) Simulated Kolmogorov distance of Z_n for Lognormal(0.6), F(5, 10) and Pareto(2.5), the parameters of
    version 1, against the leading term |skewness| phi(0) / (6 sqrt n) where the skewness exists, and the
    mean noise level for R = 100,000. Slopes are fitted for n >= 10 and only where the simulated distance exceeds
    three times the noise level.
"""

import math

import numpy as np
from scipy import special

from common import SEED, Distribution, name, save_table
from cltspeed import leading_term
from cltspeed.montecarlo import ks_exact, noise_floor, sample_means
from cltspeed.style import AQUA, BLUE, GOLD, MUTED, ORANGE, PINK, L, save_figure, use_style

R = 100_000
R_IQR_ONLY = 20_000
NS = np.unique(np.round(np.geomspace(1, 10_000, 25)).astype(int))
noise = noise_floor(R)[0]

dists = {"Lognormal(0.6)": Distribution("Lognormal"), "F(5, 10)": Distribution("F"),
         "Pareto(2.5)": Distribution("Pareto"), "Pareto(1.5)": Distribution("Pareto", {"al": 1.5}),
         "Cauchy": Distribution("Cauchy")}
means = {}
for i, (label, dist) in enumerate(dists.items()):
    reps = R_IQR_ONLY if label in ("Pareto(1.5)", "Cauchy") else R
    means[label] = sample_means(dist, NS, reps, np.random.default_rng(SEED + i))
    print(f"simulated {label}")

Q = special.ndtri(0.75)
iqr_rows, ks_rows = [], []
for j, n in enumerate(NS):
    row = {"n": int(n), "Normal_exact": 2 * Q / math.sqrt(n)}
    for label in ("Pareto(2.5)", "Pareto(1.5)", "Cauchy"):
        q1, q3 = np.percentile(means[label][j], [25, 75])
        row[label] = q3 - q1
    iqr_rows.append(row)
    row = {"n": int(n)}
    for label in ("Lognormal(0.6)", "F(5, 10)", "Pareto(2.5)"):
        d = dists[label]
        z = math.sqrt(n) * (means[label][j] - d.mean()) / d.sd()
        row[label] = ks_exact(z)
        lead = leading_term(d, int(n))
        row[f"{label} leading term"] = lead[0] if lead else ""
    ks_rows.append(row)
save_table(iqr_rows, "table7_iqr")
save_table(ks_rows, "table8_simulated_distance")


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


summary = []
for label in ("Pareto(2.5)", "Pareto(1.5)", "Cauchy"):
    sel = [r for r in iqr_rows if r["n"] >= 100]
    summary.append({"quantity": "IQR", "distribution": label,
                    "slope": slope([r["n"] for r in sel], [r[label] for r in sel]),
                    "value_at_1": iqr_rows[0][label], "value_at_10000": iqr_rows[-1][label]})
for label in ("Lognormal(0.6)", "F(5, 10)", "Pareto(2.5)"):
    sel = [r for r in ks_rows if r["n"] >= 10 and r[label] > 3 * noise]
    entry = {"quantity": "simulated distance", "distribution": label,
             "slope": slope([r["n"] for r in sel], [r[label] for r in sel]) if len(sel) >= 3 else "",
             "value_at_1": ks_rows[0][label], "value_at_10000": ks_rows[-1][label],
             "fit_n_min": sel[0]["n"] if sel else "", "fit_n_max": sel[-1]["n"] if sel else ""}
    for n in (10, 100, 1000):
        r = next(r for r in ks_rows if r["n"] >= n)
        entry[f"simulated_at_{r['n']}"] = r[label]
        entry[f"leading_at_{r['n']}"] = r[f"{label} leading term"]
    summary.append(entry)
summary.append({"quantity": "noise level", "distribution": f"R = {R}", "slope": "", "value_at_1": noise,
                "value_at_10000": noise})
keys = sorted({k for s in summary for k in s}, key=lambda k: list(summary[3].keys()).index(k)
              if k in summary[3] else 99)
save_table([{k: s.get(k, "") for k in keys} for s in summary], "table9_heavy_tails_summary")
for s in summary:
    print(s)

use_style()


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.loglog(NS, [r["Normal_exact"] for r in iqr_rows], color=MUTED, lw=1.2, ls="--",
              label=L("Normal(0, 1), exact", "Normal(0, 1), kesin"))
    for label, c, mk in (("Pareto(2.5)", BLUE, "o"), ("Pareto(1.5)", AQUA, "s"), ("Cauchy", ORANGE, "D")):
        ax.loglog(NS, [r[label] for r in iqr_rows], color=c, lw=1.2, marker=mk, ms=2.8, label=name(label))
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("IQR of the sample mean", "Örneklem ortalamasının ÇAA'sı"))
    ax.set_title(L("Spread of the sample mean", "Örneklem ortalamasının yayılımı"))
    ax.legend(loc="lower left", fontsize=7.5)
    ax = axes[1]
    ax.axhspan(1e-5, noise, color=MUTED, alpha=0.13, lw=0)
    ax.axhline(noise, color=MUTED, lw=0.8, ls=":")
    for label, c, mk in (("Lognormal(0.6)", GOLD, "^"), ("F(5, 10)", PINK, "v"), ("Pareto(2.5)", BLUE, "o")):
        ax.loglog(NS, [r[label] for r in ks_rows], color=c, lw=1.2, marker=mk, ms=2.8, label=name(label))
        lead = [r[f"{label} leading term"] for r in ks_rows]
        if lead[0] != "":
            ax.loglog(NS, lead, color=c, lw=0.8, ls="--")
    ax.text(1.2, noise * 0.8, L("mean noise, R = 100,000", "ortalama gürültü, R = 100.000"), fontsize=7.5,
            color=MUTED, va="top")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("Simulated distance", "Benzetimle uzaklık"))
    ax.set_title(L("Distance to normality (dashed: leading term)", "Normalden uzaklık (kesikli: öncü terim)"))
    ax.legend(loc="upper right", fontsize=7.5)
    return fig


save_figure("fig3_heavy_tails", draw,
            lambda: "Figure 3. Heavy tails: slow convergence, and no convergence to the normal distribution")
