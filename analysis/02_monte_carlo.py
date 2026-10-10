"""Analysis 2: what the simulated distance of version 1 measures.

Why: version 1 estimated the distance to normality from R = 2000 simulated standardized means. With finite R
the empirical distribution function differs from the true one even when Z_n is exactly normal; the mean of
that noise is about 0.8687 / sqrt(R) (0.0193 for R = 2000). Once the exact distance D_n is below this level,
the simulation shows noise, not convergence.

Design: the simulation of version 1 is repeated draw for draw (seed 42, R = 2000, 30 sample sizes from 10 to
10^4) for the Normal, Uniform and Exponential distributions, and compared with the exact distances. The
noise level is the exact mean of the Kolmogorov statistic of 2000 draws; it is also simulated for the grid
distance used by version 1 (400 samples of 2000 normal draws). Finally, the number of replications at which
the noise level equals D_n is computed for the six distributions of analysis 1.
"""

import math

import numpy as np

from common import EXACT, Distribution, name, read_table, save_table
from cltspeed import kolmogorov_distance
from cltspeed.montecarlo import KOLMOGOROV_MEAN, ks_grid_v1, noise_floor, replications_needed, v1_race
from cltspeed.style import AQUA, BLUE, GOLD, MUTED, ORANGE, PINK, PURPLE, L, save_figure, use_style

R = 2000
floor_mean, floor_q95 = noise_floor(R)
grid_floor = np.array([ks_grid_v1(np.random.default_rng(1000 + i).standard_normal(R)) for i in range(400)])

SHOWN = [Distribution("Normal"), Distribution("Uniform"), Distribution("Exponential")]
ALL_V1 = [Distribution(n) for n in ("Normal", "Uniform", "Exponential", "Laplace", "Lognormal", "Chi-square",
                                    "F", "Student-t", "Pareto")]
races = {d.label: v1_race(d) for d in ALL_V1}

rows = []
for dist in SHOWN:
    for r in races[dist.label]:
        exact = 0.0 if dist.name == "Normal" else kolmogorov_distance(dist, r["n"])[0]
        rows.append({"distribution": dist.label, "n": r["n"], "ks_v1": r["ks_grid_v1"], "exact_D_n": exact})
save_table(rows, "table3_v1_race")

# For each distribution of version 1 with an exact distance: the number of its 30 frames in which the exact
# distance was already below the mean noise of the grid distance of version 1, and the largest simulated distance (version 1 drew the
# difference curve with fixed limits of about +-0.055, so larger values were cut off).
frames = []
for dist in ALL_V1:
    rr = races[dist.label]
    row = {"distribution": dist.label, "frames": len(rr),
           "max_ks_v1": max(r["ks_grid_v1"] for r in rr),
           "frames_ks_above_0.055": sum(r["ks_grid_v1"] > 0.055 for r in rr),
           "max_share_beyond_12": max(r["tail_share"] for r in rr),
           "ks_v1_at_10000": rr[-1]["ks_grid_v1"]}
    if dist.has_cf() or dist.name in ("Exponential", "Chi-square"):
        ex = [0.0 if dist.name == "Normal" else kolmogorov_distance(dist, r["n"])[0] for r in rr]
        row["frames_exact_below_noise"] = sum(e < grid_floor.mean() for e in ex)
    else:
        row["frames_exact_below_noise"] = ""
    frames.append(row)
save_table(frames, "table4_v1_frames")

needed = []
for dist in EXACT:
    for n in (10, 30, 100, 1000, 10000):
        d = kolmogorov_distance(dist, n)[0]
        needed.append({"distribution": dist.label, "n": n, "D_n": d, "replications_needed": float(replications_needed(d))})
save_table(needed, "table5_replications")

save_table([{"R": R, "noise_mean_exact": floor_mean, "noise_q95_exact": floor_q95,
             "noise_mean_asymptotic": KOLMOGOROV_MEAN / math.sqrt(R),
             "noise_mean_v1_grid_simulated": float(grid_floor.mean()),
             "noise_mean_v1_grid_se": float(grid_floor.std(ddof=1) / math.sqrt(grid_floor.size))}],
           "table6_noise")
print(f"noise: exact mean {floor_mean:.5f}, q95 {floor_q95:.5f}, v1 grid {grid_floor.mean():.5f}")

summary = {r["distribution"]: r for r in read_table("table1_exact_distance")}
use_style()


def draw():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    ax = axes[0]
    from matplotlib.lines import Line2D
    noise = float(grid_floor.mean())
    ax.axhspan(1e-7, noise, color=MUTED, alpha=0.13, lw=0)
    ax.axhline(noise, color=MUTED, lw=0.8, ls=":")
    handles = []
    names = {"Normal": L("Normal", "Normal"), "Uniform": L("Uniform", "Tekdüze"),
             "Exponential": L("Exponential", "Üstel")}
    for dist, c, mk in zip(SHOWN, (MUTED, BLUE, ORANGE), ("o", "s", "D")):
        sub = [r for r in rows if r["distribution"] == dist.label]
        ax.loglog([r["n"] for r in sub], [r["ks_v1"] for r in sub], ls="none", marker=mk, ms=3, color=c)
        if dist.name != "Normal":
            nn = np.unique(np.round(np.geomspace(10, 10_000, 40)).astype(int))
            ax.loglog(nn, [kolmogorov_distance(dist, int(k))[0] for k in nn], color=c, lw=1.2)
        handles.append(Line2D([], [], color=c, marker=mk, ms=3, lw=0 if dist.name == "Normal" else 1.2,
                              label=names[dist.name]))
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=":", label=L("Mean noise, R = 2,000",
                                                                         "Ortalama gürültü, R = 2.000")))
    ax.set_ylim(5e-5, 1.0)
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("Distance to normality", "Normalden uzaklık"))
    ax.set_title(L("Version 1 simulation against exact $D_n$", "1. sürüm benzetimi ve kesin $D_n$"))
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax = axes[1]
    for dist, c, mk in zip(EXACT, (BLUE, AQUA, PURPLE, PINK, GOLD, ORANGE), ("o", "s", "^", "v", "D", "P")):
        sub = [r for r in read_table("table1_exact_distance") if r["distribution"] == dist.label and int(r["n"]) >= 10]
        ns = np.array([int(r["n"]) for r in sub])
        ds = np.array([float(r["D_n"]) for r in sub])
        ax.loglog(ns, replications_needed(ds), color=c, lw=1.2, marker=mk, ms=2.6, markevery=3, label=name(dist.label))
    for value, text in ((2000, L("version 1 default", "1. sürüm varsayılanı")),
                        (6000, L("version 1 maximum", "1. sürüm üst sınırı"))):
        ax.axhline(value, color=MUTED, lw=0.8, ls=":")
        ax.text(9000, value * 1.3, text, fontsize=7.5, color=MUTED, ha="right")
    ax.set_xlabel(L("Sample size n", "Örneklem büyüklüğü n"))
    ax.set_ylabel(L("Replications R", "Tekrar sayısı R"))
    ax.set_title(L("R at which the noise equals $D_n$", "Gürültünün $D_n$'ye eşit olduğu R"))
    ax.legend(loc="upper left", fontsize=7)
    return fig


save_figure("fig2_monte_carlo", draw,
            lambda: "Figure 2. What a simulation with R = 2,000 replications can and cannot measure")
