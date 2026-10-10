"""The ten population distributions of version 1, with their moments and, where it has a closed form, the
characteristic function of the standardized variable Y = (X - mu) / sigma.

Parameters and samplers are those of version 1, so that its simulations can be reproduced draw for draw:
each sampler calls the same NumPy Generator method with the same arguments.

Moments that do not exist (infinite or undefined) are returned as None.

Author: Muhammed Ikbal Yilmaz
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import special

# Default parameters of version 1 (its widgets). Chi-square and Student-t shared one "df" widget, default 5.
V1_DEFAULTS = {
    "Normal": {"mu": 0.0, "sg": 1.0},
    "Uniform": {"a": 0.0, "b": 1.0},
    "Exponential": {"sc": 1.0},
    "Laplace": {"mu": 0.0, "b": 1.0},
    "Lognormal": {"ml": 0.0, "sl": 0.6},
    "Chi-square": {"df": 5.0},
    "F": {"d1": 5.0, "d2": 10.0},
    "Student-t": {"df": 5.0},
    "Pareto": {"xm": 1.0, "al": 2.5},
    "Cauchy": {},
}
NAMES = tuple(V1_DEFAULTS)


@dataclass
class Distribution:
    name: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in V1_DEFAULTS:
            raise ValueError(f"unknown distribution {self.name!r}; choose from {NAMES}")
        self.params = {**V1_DEFAULTS[self.name], **self.params}
        self._check()

    # ------------------------------------------------------------------------------------------- checks
    def _check(self):
        p = self.params
        positive = {"Normal": ["sg"], "Exponential": ["sc"], "Laplace": ["b"], "Lognormal": ["sl"],
                    "Chi-square": ["df"], "F": ["d1", "d2"], "Student-t": ["df"], "Pareto": ["xm", "al"]}
        for key in positive.get(self.name, []):
            if not p[key] > 0:
                raise ValueError(f"{self.name}: parameter {key} must be positive")
        if self.name == "Uniform" and not p["b"] > p["a"]:
            raise ValueError("Uniform: b must be larger than a")

    @property
    def label(self):
        p = self.params
        if self.name == "Chi-square":
            return f"Chi-square({p['df']:g})"
        if self.name == "Student-t":
            return f"Student-t({p['df']:g})"
        if self.name == "Pareto":
            return f"Pareto({p['al']:g})"
        if self.name == "F":
            return f"F({p['d1']:g}, {p['d2']:g})"
        if self.name == "Lognormal":
            return f"Lognormal({p['sl']:g})"
        return self.name

    # ------------------------------------------------------------------------------------------ sampler
    def sample(self, rng, shape):
        """Draws with exactly the Generator calls of version 1."""
        p, d = self.params, self.name
        if d == "Normal":
            return rng.normal(p["mu"], p["sg"], size=shape)
        if d == "Uniform":
            return rng.uniform(p["a"], p["b"], size=shape)
        if d == "Exponential":
            return rng.exponential(p["sc"], size=shape)
        if d == "Laplace":
            return rng.laplace(p["mu"], p["b"], size=shape)
        if d == "Lognormal":
            return rng.lognormal(p["ml"], p["sl"], size=shape)
        if d == "Chi-square":
            return rng.chisquare(p["df"], size=shape)
        if d == "F":
            return rng.f(p["d1"], p["d2"], size=shape)
        if d == "Student-t":
            return rng.standard_t(p["df"], size=shape)
        if d == "Pareto":
            # NumPy's pareto() is the Lomax distribution; xm * (1 + Lomax) is the classical Pareto with
            # minimum xm and tail index al.
            return p["xm"] * (1 + rng.pareto(p["al"], size=shape))
        return rng.standard_t(1.0, size=shape)  # Cauchy = Student-t with one degree of freedom

    # ------------------------------------------------------------------------------------------ moments
    def mean(self):
        p, d = self.params, self.name
        if d == "Normal":
            return p["mu"]
        if d == "Uniform":
            return 0.5 * (p["a"] + p["b"])
        if d == "Exponential":
            return p["sc"]
        if d == "Laplace":
            return p["mu"]
        if d == "Lognormal":
            return math.exp(p["ml"] + 0.5 * p["sl"] ** 2)
        if d == "Chi-square":
            return p["df"]
        if d == "F":
            return p["d2"] / (p["d2"] - 2) if p["d2"] > 2 else None
        if d == "Student-t":
            return 0.0 if p["df"] > 1 else None
        if d == "Pareto":
            a = p["al"]
            return a * p["xm"] / (a - 1) if a > 1 else None
        return None

    def sd(self):
        p, d = self.params, self.name
        if d == "Normal":
            return p["sg"]
        if d == "Uniform":
            return (p["b"] - p["a"]) / math.sqrt(12)
        if d == "Exponential":
            return p["sc"]
        if d == "Laplace":
            return math.sqrt(2) * p["b"]
        if d == "Lognormal":
            s2 = p["sl"] ** 2
            return math.sqrt(math.expm1(s2) * math.exp(2 * p["ml"] + s2))
        if d == "Chi-square":
            return math.sqrt(2 * p["df"])
        if d == "F":
            d1, d2 = p["d1"], p["d2"]
            if d2 <= 4:
                return None
            return math.sqrt(2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4)))
        if d == "Student-t":
            v = p["df"]
            return math.sqrt(v / (v - 2)) if v > 2 else None
        if d == "Pareto":
            a, xm = p["al"], p["xm"]
            return math.sqrt(a * xm ** 2 / ((a - 1) ** 2 * (a - 2))) if a > 2 else None
        return None

    def skewness(self):
        p, d = self.params, self.name
        if d in ("Normal", "Uniform", "Laplace"):
            return 0.0
        if d == "Exponential":
            return 2.0
        if d == "Lognormal":
            e = math.exp(p["sl"] ** 2)
            return (e + 2) * math.sqrt(e - 1)
        if d == "Chi-square":
            return math.sqrt(8 / p["df"])
        if d == "F":
            d1, d2 = p["d1"], p["d2"]
            if d2 <= 6:
                return None
            return (2 * d1 + d2 - 2) * math.sqrt(8 * (d2 - 4)) / ((d2 - 6) * math.sqrt(d1 * (d1 + d2 - 2)))
        if d == "Student-t":
            return 0.0 if p["df"] > 3 else None
        if d == "Pareto":
            a = p["al"]
            return 2 * (1 + a) / (a - 3) * math.sqrt((a - 2) / a) if a > 3 else None
        return None

    def excess_kurtosis(self):
        p, d = self.params, self.name
        if d == "Normal":
            return 0.0
        if d == "Uniform":
            return -1.2
        if d == "Exponential":
            return 6.0
        if d == "Laplace":
            return 3.0
        if d == "Lognormal":
            s2 = p["sl"] ** 2
            return math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 6
        if d == "Chi-square":
            return 12 / p["df"]
        if d == "F":
            d1, d2 = p["d1"], p["d2"]
            if d2 <= 8:
                return None
            num = d1 * (5 * d2 - 22) * (d1 + d2 - 2) + (d2 - 4) * (d2 - 2) ** 2
            return 12 * num / (d1 * (d2 - 6) * (d2 - 8) * (d1 + d2 - 2))
        if d == "Student-t":
            v = p["df"]
            return 6 / (v - 4) if v > 4 else None
        if d == "Pareto":
            a = p["al"]
            return 6 * (a ** 3 + a ** 2 - 6 * a - 2) / (a * (a - 3) * (a - 4)) if a > 4 else None
        return None

    def clt_applies(self):
        """True when the mean and a positive, finite variance exist (Lindeberg-Levy CLT)."""
        return self.mean() is not None and self.sd() is not None

    # ------------------------------------------------------------------------- characteristic function
    def has_cf(self):
        return self.name in ("Normal", "Uniform", "Exponential", "Laplace", "Chi-square") or (
            self.name == "Student-t" and self.params["df"] > 2)

    def cf_sum(self, t, n):
        """Characteristic function of Z_n = (Y_1 + ... + Y_n) / sqrt(n), Y = (X - mu) / sigma, at t >= 0.

        Computed as exp(n log phi_Y(t / sqrt(n))) wherever phi_Y is positive, which keeps the result accurate
        for large n. Returns a complex array.
        """
        if not self.has_cf():
            raise ValueError(f"{self.label}: no closed-form characteristic function")
        t = np.asarray(t, dtype=float)
        s = t / math.sqrt(n)
        d = self.name
        if d == "Normal":
            return np.exp(-0.5 * t * t).astype(complex)
        if d == "Uniform":
            u = math.sqrt(3) * s
            sinc = np.where(u == 0, 1.0, np.sin(u) / np.where(u == 0, 1.0, u))
            return np.power(sinc, n).astype(complex)  # can be negative: integer power, not logarithms
        if d == "Laplace":
            return np.exp(-n * np.log1p(0.5 * s * s)).astype(complex)
        if d in ("Exponential", "Chi-square"):
            a = 1.0 if d == "Exponential" else 0.5 * self.params["df"]  # Gamma shape
            z = s / math.sqrt(a)
            # log phi_Y(s) = -a/2 log(1 + z^2) + i a (arctan z - z), Y = (G - a) / sqrt(a), G ~ Gamma(a, 1)
            log_phi = -0.5 * a * np.log1p(z * z) + 1j * a * (np.arctan(z) - z)
            return np.exp(n * log_phi)
        # Student-t with df > 2: phi_X(t) = K_{v/2}(sqrt(v)|t|) (sqrt(v)|t|)^{v/2} / (Gamma(v/2) 2^{v/2-1})
        v = self.params["df"]
        x = math.sqrt(v) * np.abs(s) / math.sqrt(v / (v - 2))  # argument for Y = X / sd
        half = 0.5 * v
        with np.errstate(divide="ignore", invalid="ignore"):
            log_phi = (np.log(special.kve(half, x)) - x + half * np.log(x)
                       - special.gammaln(half) - (half - 1) * math.log(2))
        log_phi = np.where(x < 1e-12, 0.0, log_phi)
        return np.exp(n * log_phi).astype(complex)


def v1_distributions():
    """All ten distributions with the default parameters of version 1."""
    return [Distribution(name) for name in NAMES]
