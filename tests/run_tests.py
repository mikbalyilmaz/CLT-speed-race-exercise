"""Unit tests. Run from the repository root:  python tests/run_tests.py

Each result is compared with an independent computation: SciPy's distribution objects, numerical Fourier
integrals of the densities, exact rational arithmetic, closed-form distribution functions, a brute-force
search, or the output of version 1.
"""

import math
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import integrate, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cltspeed import HE3_MAX, PHI0, Distribution, GilPelaez, irwin_hall_cdf, kolmogorov_distance  # noqa: E402
from cltspeed import leading_term, t3_constant  # noqa: E402
from cltspeed.exact import _gamma_difference, _irwin_hall_difference  # noqa: E402
from cltspeed.montecarlo import KOLMOGOROV_MEAN, ks_exact, noise_floor, sample_means, v1_race  # noqa: E402

failed = 0


def check(name, ok):
    global failed
    print(f"{'ok' if ok else 'FAIL':<5}{name}")
    if not ok:
        failed += 1


def close(a, b, tol):
    return a is not None and b is not None and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(b))


# Moments against scipy.stats ---------------------------------------------------------------------------
cases = [
    (Distribution("Normal"), stats.norm(0, 1)),
    (Distribution("Uniform"), stats.uniform(0, 1)),
    (Distribution("Exponential"), stats.expon(scale=1)),
    (Distribution("Laplace"), stats.laplace(0, 1)),
    (Distribution("Lognormal"), stats.lognorm(s=0.6, scale=1)),
    (Distribution("Chi-square"), stats.chi2(5)),
    (Distribution("F"), stats.f(5, 10)),
    (Distribution("F", {"d1": 5, "d2": 12}), stats.f(5, 12)),
    (Distribution("Student-t"), stats.t(5)),
    (Distribution("Student-t", {"df": 7}), stats.t(7)),
    (Distribution("Pareto", {"al": 5.0}), stats.pareto(b=5.0, scale=1)),
]
ok = True
for dist, ref in cases:
    m, v, s, k = (float(x) for x in ref.stats(moments="mvsk"))
    ok &= close(dist.mean(), m, 1e-12) and close(dist.sd(), math.sqrt(v), 1e-12)
    ok &= close(dist.skewness(), s, 1e-10) and close(dist.excess_kurtosis(), k, 1e-9)
check("mean, sd, skewness and excess kurtosis equal scipy.stats for 11 parameter settings", ok)

check("moments that do not exist are reported as missing (Pareto 2.5, Student-t 3, F(5,10) kurtosis, Cauchy)",
      Distribution("Pareto").skewness() is None and Distribution("Pareto").sd() is not None
      and Distribution("Student-t", {"df": 3}).excess_kurtosis() is None
      and Distribution("Pareto", {"al": 1.5}).sd() is None and Distribution("Cauchy").mean() is None
      and Distribution("F").excess_kurtosis() is not None and Distribution("F", {"d2": 8}).excess_kurtosis() is None)

rng = np.random.default_rng(1)
ok = True
for name in ("Uniform", "Exponential", "Laplace", "Lognormal", "Chi-square", "F", "Student-t"):
    d = Distribution(name)
    x = d.sample(rng, 400_000)
    ok &= abs(x.mean() - d.mean()) < 5 * d.sd() / math.sqrt(x.size)
check("sample means of 400,000 draws are within 5 standard errors of the theoretical means", ok)


# Characteristic functions against numerical Fourier integrals of the densities --------------------------
def numeric_cf(dist, ref, t):
    mu, sd = dist.mean(), dist.sd()
    lo, hi = ref.support()

    def dens(y):
        return sd * ref.pdf(mu + sd * y)

    a = (lo - mu) / sd if math.isfinite(lo) else -np.inf
    b = (hi - mu) / sd if math.isfinite(hi) else np.inf
    parts = [(a, 0.0), (0.0, b)] if a < 0 < b else [(a, b)]
    re = im = 0.0
    for p, q in parts:
        if math.isfinite(p) and math.isfinite(q):
            re += integrate.quad(dens, p, q, weight="cos", wvar=t, limit=400)[0]
            im += integrate.quad(dens, p, q, weight="sin", wvar=t, limit=400)[0]
        elif math.isinf(q):
            re += integrate.quad(dens, p, np.inf, weight="cos", wvar=t, limit=400)[0]
            im += integrate.quad(dens, p, np.inf, weight="sin", wvar=t, limit=400)[0]
        else:
            re += integrate.quad(lambda y: dens(-y), -q, np.inf, weight="cos", wvar=t, limit=400)[0]
            im -= integrate.quad(lambda y: dens(-y), -q, np.inf, weight="sin", wvar=t, limit=400)[0]
    return complex(re, im)


cf_cases = [(Distribution("Uniform"), stats.uniform(0, 1)), (Distribution("Laplace"), stats.laplace(0, 1)),
            (Distribution("Exponential"), stats.expon()), (Distribution("Chi-square"), stats.chi2(5)),
            (Distribution("Student-t", {"df": 3}), stats.t(3)), (Distribution("Student-t"), stats.t(5)),
            (Distribution("Student-t", {"df": 7}), stats.t(7))]
err = max(abs(d.cf_sum(np.array([t]), 1)[0] - numeric_cf(d, ref, t)) for d, ref in cf_cases for t in (0.3, 1.0, 2.5))
check(f"characteristic functions equal numerical Fourier integrals of the densities (max error {err:.1e})",
      err < 1e-7)

s = np.linspace(0, 6, 61)
t3 = Distribution("Student-t", {"df": 3}).cf_sum(s, 1).real
t5 = Distribution("Student-t").cf_sum(s, 1).real
check("Student-t: the Bessel formula equals the closed forms (1+|s|)e^-|s| (df 3) and "
      "(1+sqrt3|s|+s^2)e^-sqrt3|s| (df 5)",
      np.max(np.abs(t3 - (1 + s) * np.exp(-s))) < 1e-12
      and np.max(np.abs(t5 - (1 + math.sqrt(3) * s + s * s) * np.exp(-math.sqrt(3) * s))) < 1e-12)

d = Distribution("Exponential")
check("phi_n(t) = phi_1(t / sqrt n)^n",
      np.max(np.abs(d.cf_sum(s, 7) - d.cf_sum(s / math.sqrt(7), 1) ** 7)) < 1e-13)

# Irwin-Hall in exact arithmetic --------------------------------------------------------------------------
check("Irwin-Hall: F = s for n = 1, triangle formula for n = 2, F(n/2) = 1/2, F(n) = 1",
      irwin_hall_cdf(Fraction(1, 3), 1) == Fraction(1, 3)
      and irwin_hall_cdf(Fraction(3, 2), 2) == 1 - Fraction(1, 2) * Fraction(1, 2) ** 2
      and irwin_hall_cdf(Fraction(7, 2), 7) == Fraction(1, 2) and irwin_hall_cdf(9, 9) == 1)

# Gil-Pelaez inversion against exact distribution functions --------------------------------------------
x = np.linspace(-6, 6, 241)
errs = {}
for n in (5, 50):
    errs[f"Exponential n={n}"] = np.max(np.abs(GilPelaez(Distribution("Exponential"), n)(x) - _gamma_difference(n, x)))
    errs[f"Chi-square(5) n={n}"] = np.max(np.abs(GilPelaez(Distribution("Chi-square"), n)(x)
                                                 - _gamma_difference(2.5 * n, x)))
for n in (12, 30):
    errs[f"Uniform n={n}"] = np.max(np.abs(GilPelaez(Distribution("Uniform"), n)(x) - _irwin_hall_difference(n, x)))
# Laplace(0,1), n = 2: S = L1 + L2 has P(S <= s) = 1 - (2 + s) e^-s / 4 for s >= 0; Z_2 = S / 2
s2 = 2 * np.abs(x)
upper = 1 - (2 + s2) * np.exp(-s2) / 4
lap2 = np.where(x >= 0, upper, 1 - upper) - special.ndtr(x)
errs["Laplace n=2"] = np.max(np.abs(GilPelaez(Distribution("Laplace"), 2)(x) - lap2))
for v in (3, 5):
    exact = special.stdtr(v, x * math.sqrt(v / (v - 2))) - special.ndtr(x)
    errs[f"Student-t({v}) n=1"] = np.max(np.abs(GilPelaez(Distribution("Student-t", {"df": v}), 1)(x) - exact))
worst = max(errs, key=errs.get)
check(f"Gil-Pelaez inversion equals exact distribution functions in 9 cases (largest error {errs[worst]:.1e}, "
      f"{worst})", errs[worst] < 1e-10)

# Supremum search against a brute-force grid ---------------------------------------------------------------
ok = True
for name, n in (("Exponential", 10), ("Chi-square", 3), ("Uniform", 7)):
    dist = Distribution(name)
    D, _, _ = kolmogorov_distance(dist, n)
    fine = np.arange(-8, 8, 2e-4)
    if name == "Uniform":
        fine = np.arange(-3, 3, 2e-3)
        brute = np.max(np.abs(_irwin_hall_difference(n, fine)))
    else:
        brute = np.max(np.abs(_gamma_difference(n * (1.0 if name == "Exponential" else 2.5), fine)))
    ok &= brute <= D + 1e-12 and D - brute < 1e-7
check("D_n is at least the maximum over a brute-force grid and exceeds it by less than 1e-7", ok)

# Leading terms ---------------------------------------------------------------------------------------------
xx = np.linspace(0, 5, 500_001)
check("max |He_3(x) phi(x)| and max |He_2(x) phi(x)| equal their closed forms",
      abs(np.max(np.abs((xx ** 3 - 3 * xx) * np.exp(-xx * xx / 2) / math.sqrt(2 * math.pi))) - HE3_MAX) < 1e-10
      and abs(np.max(np.abs((xx ** 2 - 1) * np.exp(-xx * xx / 2) / math.sqrt(2 * math.pi))) - PHI0) < 1e-12)
ratios = [kolmogorov_distance(Distribution(nm, pr), 10_000)[0] / leading_term(Distribution(nm, pr), 10_000)[0]
          for nm, pr in (("Exponential", {}), ("Uniform", {}), ("Laplace", {}), ("Student-t", {"df": 3}))]
check("D_n / leading term at n = 10^4 is within 0.3% of 1 (Exponential, Uniform, Laplace, Student-t(3))",
      max(abs(r - 1) for r in ratios) < 3e-3)
check("Student-t(3) constant equals a direct quadrature at its maximiser",
      abs(t3_constant() - abs(integrate.quad(lambda t: t * t * math.exp(-t * t / 2), 0, np.inf, weight="sin",
                                             wvar=0.8424178)[0]) / (3 * math.pi)) < 1e-9)

# Monte Carlo ------------------------------------------------------------------------------------------------
z = np.random.default_rng(3).standard_normal(500)
zs = np.sort(z)
cdf = special.ndtr(zs)
brute = max(np.max(np.abs(np.arange(1, 501) / 500 - cdf)), np.max(np.abs(np.arange(0, 500) / 500 - cdf)))
check("empirical Kolmogorov distance equals the brute-force value at the jumps", abs(ks_exact(z) - brute) < 1e-15)

mean, _ = noise_floor(2000)
sims = [ks_exact(np.random.default_rng(100 + i).standard_normal(2000)) for i in range(400)]
check(f"noise level for R = 2000: exact mean {mean:.5f}; simulated {np.mean(sims):.5f}; "
      f"asymptotic {KOLMOGOROV_MEAN / math.sqrt(2000):.5f}",
      abs(np.mean(sims) - mean) < 3 * np.std(sims) / math.sqrt(400)
      and abs(mean * math.sqrt(2000) - KOLMOGOROV_MEAN) < 0.01)

a = sample_means(Distribution("Exponential"), [5, 50], 3000, np.random.default_rng(9), chunk=1000)
b = sample_means(Distribution("Exponential"), [5, 50], 3000, np.random.default_rng(9), chunk=1000)
check("sample_means is reproducible and its means are within 4 standard errors of mu",
      np.array_equal(a, b) and abs(a[1].mean() - 1) < 4 / math.sqrt(50 * 3000))

# Values printed in the frames that version 1 saved in its repository (seed 42, R = 2000)
v1 = {("Normal", 10000): 0.0151, ("Exponential", 10000): 0.0173, ("Uniform", 10): 0.0101, ("Pareto", 10): 0.1510}
ok = True
for (name, n), value in v1.items():
    row = next(r for r in v1_race(Distribution(name)) if r["n"] == n)
    ok &= round(row["ks_grid_v1"], 4) == value
check("the reimplementation reproduces four KS values printed in the frames of version 1", ok)

# The race: three frames and a GIF, without Jupyter
from cltspeed import race  # noqa: E402

tmp = Path(tempfile.mkdtemp())
try:
    rows, path = race.run(Distribution("Uniform"), R=400, nmax=40, steps=3, seed=1, outdir=tmp, fmt="gif")
    check("the race runs without Jupyter and writes its frames and a GIF",
          len(rows) == 3 and path is not None and path.exists() and len(list((tmp / "frames").glob("*.png"))) == 3)
finally:
    shutil.rmtree(tmp)

print(f"\n{failed} check(s) failed.")
sys.exit(1 if failed else 0)
