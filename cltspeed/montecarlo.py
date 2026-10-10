"""Monte Carlo estimation of the distance to normality, as in version 1, and the noise that limits it.

Version 1 simulated R paths, extended each path from one sample size to the next, and measured
max |F_hat(x) - Phi(x)| over a grid of 401 points on [-12, 12], where F_hat is the empirical distribution
function of the R standardized means. `v1_race` reproduces that computation draw for draw.

Even if Z_n were exactly normal, F_hat would differ from Phi because R is finite: the exact Kolmogorov
statistic of R draws from the true distribution has mean about 0.8687 / sqrt(R) (`noise_floor`). Distances
smaller than this cannot be seen in a simulation with R replications.

Author: Muhammed Ikbal Yilmaz
"""

import math

import numpy as np
from scipy import stats

from .exact import normal_cdf

KOLMOGOROV_MEAN = math.sqrt(math.pi / 2) * math.log(2)  # mean of the limiting Kolmogorov distribution


def v1_sample_sizes(nmax=10_000, steps=30, n0=10):
    """The sequence of sample sizes of version 1."""
    return np.unique(np.round(np.exp(np.linspace(np.log(n0), np.log(nmax), steps))).astype(int))


def ks_exact(z):
    """sup_x |F_hat(x) - Phi(x)| for the empirical distribution function of z, attained at a data point."""
    z = np.sort(np.asarray(z, dtype=float))
    m = z.size
    cdf = normal_cdf(z)
    i = np.arange(1, m + 1)
    return float(max(np.max(i / m - cdf), np.max(cdf - (i - 1) / m)))


V1_GRID = np.linspace(-12.0, 12.0, 401)


def ks_grid_v1(z):
    """The distance of version 1: max |F_hat - Phi| over 401 grid points on [-12, 12]."""
    z = np.sort(np.asarray(z, dtype=float))
    fhat = np.searchsorted(z, V1_GRID, side="right") / z.size
    return float(np.max(np.abs(fhat - normal_cdf(V1_GRID))))


def v1_race(dist, seed=42, R=2000, nmax=10_000, steps=30):
    """Repeats the simulation of version 1 (same draws, same order) and returns one row per sample size with
    the grid distance of version 1, the exact empirical distance and the share of |Z_n| > 12."""
    rng = np.random.default_rng(int(seed))
    mu, sd = dist.mean(), dist.sd()
    s = np.zeros(R)
    nprev = 0
    rows = []
    for n in v1_sample_sizes(nmax, steps):
        x = dist.sample(rng, (R, n - nprev))
        s += x.sum(1)
        nprev = n
        z = math.sqrt(n) * (s / n - mu) / sd
        z = z[np.isfinite(z)]
        if z.size < 200:
            continue
        rows.append({"n": int(n), "ks_grid_v1": ks_grid_v1(z), "ks_exact": ks_exact(z),
                     "tail_share": float(np.mean(np.abs(z) > 12.0))})
    return rows


def sample_means(dist, ns, R, rng, chunk=2_000):
    """Sample means of R independent paths at each n in the increasing sequence ns.

    Returns an array of shape (len(ns), R). Paths are simulated in blocks of `chunk` so that memory stays
    bounded; each path is extended from one n to the next, as in version 1.
    """
    ns = np.asarray(ns, dtype=int)
    out = np.empty((ns.size, R))
    for start in range(0, R, chunk):
        rows = min(chunk, R - start)
        s = np.zeros(rows)
        nprev = 0
        for j, n in enumerate(ns):
            remaining = n - nprev
            while remaining > 0:  # bounded block of columns
                k = min(remaining, max(1, 20_000_000 // rows))
                s += dist.sample(rng, (rows, k)).sum(1)
                remaining -= k
            nprev = n
            out[j, start:start + rows] = s / n
    return out


def noise_floor(R):
    """Mean and 95% quantile of the exact Kolmogorov statistic of R draws (scipy.stats.kstwo)."""
    law = stats.kstwo(int(R))
    return float(law.mean()), float(law.ppf(0.95))


def replications_needed(distance):
    """R for which the asymptotic mean noise, KOLMOGOROV_MEAN / sqrt(R), equals the given distance."""
    return (KOLMOGOROV_MEAN / np.asarray(distance, dtype=float)) ** 2
