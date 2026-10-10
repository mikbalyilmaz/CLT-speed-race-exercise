"""The exact distance between the distribution of the standardized mean and the standard normal distribution.

For Z_n = sqrt(n) (Xbar_n - mu) / sigma with distribution function F_n, the Kolmogorov distance is
D_n = sup_x |F_n(x) - Phi(x)|. It is computed without simulation:

* Exponential and Chi-square: the sum of n draws is Gamma distributed, so F_n is a regularized incomplete
  gamma function (exact).
* Uniform, n <= 40: the sum of n draws has the Irwin-Hall distribution; its distribution function is
  evaluated in exact rational arithmetic (fractions.Fraction), so there is no rounding before the final
  conversion to floating point.
* Laplace, n = 1: closed form.
* Otherwise (Laplace n >= 2, Uniform n > 40, Student-t): numerical inversion of the characteristic
  function with the Gil-Pelaez formula, applied to the difference of the two characteristic functions,
  F_n(x) - Phi(x) = (1/pi) int_0^inf [Re psi(t) sin(tx) - Im psi(t) cos(tx)] / t dt,
  psi(t) = phi_n(t) - exp(-t^2 / 2),
  with composite Gauss-Legendre quadrature. The unit tests check it against the exact methods.

The supremum is found on a grid of step 0.02 on [-8, 8] and refined around the largest local maxima.

Author: Muhammed Ikbal Yilmaz
"""

import math
from fractions import Fraction

import numpy as np
from scipy import integrate, optimize, special

X_MAX = 8.0
GRID_STEP = 0.02
IRWIN_HALL_MAX_N = 40
TAIL_TOL = 1e-13    # |psi(t)| beyond the truncation point T stays below this
T_LIMIT = 5_000.0   # larger T would make the quadrature slow; exact methods cover those cases
PANEL = 0.1         # width of each Gauss-Legendre panel in t
NODES, WEIGHTS = np.polynomial.legendre.leggauss(10)

# max_x |He_3(x) phi(x)| = |(x^3 - 3x) phi(x)|, attained at x^2 = 3 - sqrt(6)
HE3_MAX = math.sqrt(6 * (3 - math.sqrt(6))) * math.exp(-(3 - math.sqrt(6)) / 2) / math.sqrt(2 * math.pi)
PHI0 = 1 / math.sqrt(2 * math.pi)  # max_x |He_2(x) phi(x)| = phi(0)


def normal_cdf(x):
    return special.ndtr(x)


# ------------------------------------------------------------------------------------------ exact methods
def _gamma_difference(shape, x):
    """F_n(x) - Phi(x) when the standardized sum is (G - A) / sqrt(A), G ~ Gamma(A, 1)."""
    x = np.asarray(x, dtype=float)
    g = shape + x * math.sqrt(shape)
    return np.where(g > 0, special.gammainc(shape, np.maximum(g, 0.0)), 0.0) - normal_cdf(x)


def irwin_hall_cdf(s, n):
    """P(U_1 + ... + U_n <= s) for independent U(0, 1), exactly, for a Fraction (or int) s."""
    s = Fraction(s)
    if s <= 0:
        return Fraction(0)
    if s >= n:
        return Fraction(1)
    total = sum((-1) ** k * math.comb(n, k) * (s - k) ** n for k in range(int(s) + 1))
    return total / math.factorial(n)


def _irwin_hall_difference(n, x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    scale = math.sqrt(n / 12)
    exact = [float(irwin_hall_cdf(Fraction(n / 2 + xi * scale), n)) for xi in x]
    return np.array(exact) - normal_cdf(x)


def _laplace1_difference(x):
    """n = 1: Y = X / (sqrt(2) b) is Laplace with scale 1 / sqrt(2)."""
    x = np.asarray(x, dtype=float)
    r = math.sqrt(2) * x
    cdf = np.where(x < 0, 0.5 * np.exp(np.minimum(r, 0)), 1 - 0.5 * np.exp(-np.maximum(r, 0)))
    return cdf - normal_cdf(x)


# ------------------------------------------------------------------------------------------ Gil-Pelaez
def truncation_point(dist, n):
    """Smallest T on a fine logarithmic scan beyond which |psi(t)| < TAIL_TOL."""
    t = np.geomspace(1e-2, 1e6, 6000)
    psi = np.abs(dist.cf_sum(t, n) - np.exp(-0.5 * t * t))
    big = np.nonzero(psi >= TAIL_TOL)[0]
    return 1.0 if big.size == 0 else float(t[min(big[-1] + 1, t.size - 1)])


def _quadrature_nodes(T):
    panels = int(math.ceil(T / PANEL))
    left = np.arange(panels) * PANEL
    t = (left[:, None] + 0.5 * PANEL * (NODES[None, :] + 1)).ravel()
    w = np.tile(0.5 * PANEL * WEIGHTS, panels)
    return t, w


class GilPelaez:
    """F_n(x) - Phi(x) by inversion of the characteristic function; set up once per (distribution, n)."""

    def __init__(self, dist, n):
        self.T = truncation_point(dist, n)
        if self.T > T_LIMIT:
            raise ValueError(f"{dist.label}, n={n}: characteristic function decays too slowly (T={self.T:.3g})")
        t, w = _quadrature_nodes(self.T)
        psi = dist.cf_sum(t, n) - np.exp(-0.5 * t * t)
        self.t = t
        self.a = w * psi.real / t / math.pi
        self.b = w * psi.imag / t / math.pi
        self.symmetric = bool(np.all(psi.imag == 0))

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty(x.size)
        block = max(1, 4_000_000 // self.t.size)
        for i in range(0, x.size, block):
            tx = np.outer(self.t, x[i:i + block])
            val = self.a @ np.sin(tx)
            if not self.symmetric:
                val -= self.b @ np.cos(tx)
            out[i:i + block] = val
        return out


def difference_function(dist, n):
    """Returns (f, method) where f(x) = F_n(x) - Phi(x)."""
    name = dist.name
    if name == "Normal":
        return (lambda x: np.zeros(np.size(x))), "exact (normal)"
    if name == "Exponential":
        return (lambda x: _gamma_difference(n * 1.0, x)), "exact (gamma)"
    if name == "Chi-square":
        return (lambda x: _gamma_difference(n * 0.5 * dist.params["df"], x)), "exact (gamma)"
    if name == "Uniform" and n <= IRWIN_HALL_MAX_N:
        return (lambda x: _irwin_hall_difference(n, x)), "exact (Irwin-Hall, rational arithmetic)"
    if name == "Laplace" and n == 1:
        return _laplace1_difference, "exact (closed form)"
    return GilPelaez(dist, n), "Gil-Pelaez inversion"


def kolmogorov_distance(dist, n):
    """Exact Kolmogorov distance D_n between the law of Z_n and N(0, 1).

    Returns (D_n, x at which it is attained, method).
    """
    if not dist.clt_applies():
        raise ValueError(f"{dist.label}: the mean or the variance does not exist")
    f, method = difference_function(dist, n)
    grid = np.arange(-X_MAX, X_MAX + GRID_STEP / 2, GRID_STEP)
    values = np.abs(f(grid))
    if values.max() == 0:
        return 0.0, 0.0, method
    # local maxima of |F_n - Phi| on the grid, largest first
    inner = np.nonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    candidates = sorted(inner, key=lambda i: -values[i])[:3] or [int(np.argmax(values))]
    best_d, best_x = float(values.max()), float(grid[np.argmax(values)])
    for i in candidates:
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
        res = optimize.minimize_scalar(lambda z: -abs(float(f(np.array([z]))[0])), bounds=(lo, hi),
                                       method="bounded", options={"xatol": 1e-9})
        if -res.fun > best_d:
            best_d, best_x = float(-res.fun), float(res.x)
    edge = float(max(values[0], values[-1]))
    if edge > 0.1 * best_d:
        raise RuntimeError(f"{dist.label}, n={n}: the supremum may lie outside [-{X_MAX}, {X_MAX}]")
    return best_d, best_x, method


# ---------------------------------------------------------------------------------------- leading terms
_T3_CONSTANT = None


def t3_constant():
    """sqrt(n) D_n -> c for Student-t(3).

    For Y = X / sqrt(3), X ~ t(3), phi_Y(s) = (1 + |s|) exp(-|s|) = 1 - s^2/2 + |s|^3/3 + O(s^4). The term
    |s|^3/3 gives phi_n(t) = exp(-t^2/2) (1 + |t|^3 / (3 sqrt n)) + ..., and by the Gil-Pelaez formula
    F_n(x) - Phi(x) ~ I(x) / (3 pi sqrt n) with I(x) = int_0^inf t^2 exp(-t^2/2) sin(tx) dt.
    So c = max_x |I(x)| / (3 pi), computed here numerically.
    """
    global _T3_CONSTANT
    if _T3_CONSTANT is None:
        def integral(x):
            return integrate.quad(lambda t: t * t * math.exp(-0.5 * t * t), 0, np.inf, weight="sin", wvar=x)[0]
        xs = np.linspace(0.05, 4, 80)
        i = int(np.argmax([abs(integral(x)) for x in xs]))
        res = optimize.minimize_scalar(lambda x: -abs(integral(x)), bounds=(xs[i - 1], xs[i + 1]),
                                       method="bounded", options={"xatol": 1e-10})
        _T3_CONSTANT = -res.fun / (3 * math.pi)
    return _T3_CONSTANT


def leading_term(dist, n):
    """Leading term of D_n for large n, or None where this module does not provide one.

    Edgeworth expansion (finite fourth moment, continuous distribution):
    F_n(x) - Phi(x) = -phi(x) [ gamma / (6 sqrt n) He_2(x) + kappa_4 / (24 n) He_3(x)
                                 + gamma^2 / (72 n) He_5(x) ] + o(1/n),
    with gamma the skewness and kappa_4 the excess kurtosis. If gamma != 0 the leading term of D_n is
    |gamma| phi(0) / (6 sqrt n); if gamma = 0 it is |kappa_4| HE3_MAX / (24 n).
    Student-t(3) has no finite third absolute moment; its leading term is t3_constant() / sqrt(n).
    Returns (value, order, source).
    """
    if dist.name == "Student-t" and dist.params["df"] == 3:
        return t3_constant() / math.sqrt(n), 0.5, "t3"
    g = dist.skewness()
    k = dist.excess_kurtosis()
    if g is None or k is None:
        return None
    if g != 0:
        return abs(g) * PHI0 / (6 * math.sqrt(n)), 0.5, "skewness"
    if k == 0:
        return None
    return abs(k) * HE3_MAX / (24 * n), 1.0, "kurtosis"
