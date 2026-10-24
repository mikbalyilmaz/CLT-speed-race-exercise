"""cltspeed: how fast the standardized sample mean approaches the normal distribution."""

from .distributions import NAMES, V1_DEFAULTS, Distribution, v1_distributions
from .exact import (HE3_MAX, PHI0, GilPelaez, difference_function, leading_term, t3_constant, irwin_hall_cdf,
                    kolmogorov_distance, normal_cdf)

__version__ = "2.1.0"
