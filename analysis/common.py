"""Common set-up for the analysis scripts: import path, seed, CSV writer and the distributions studied."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cltspeed import Distribution  # noqa: E402
from cltspeed.style import FIG_DIR  # noqa: E402

SEED = 20261010
THRESHOLD = 0.01

# Distributions whose exact distance can be computed (analysis 1), in the order used in the figures.
# Student-t(5) and Chi-square(5) are the defaults of version 1; Student-t(3) is added as the smallest integer
# degrees of freedom with finite variance.
EXACT = [Distribution("Uniform"), Distribution("Laplace"), Distribution("Student-t"),
         Distribution("Student-t", {"df": 3}), Distribution("Chi-square"), Distribution("Exponential")]


def save_table(rows, name):
    """Write a list of dicts as figures/<name>.csv."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIG_DIR / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote figures/{name}.csv ({len(rows)} rows)")


def read_table(name):
    with open(FIG_DIR / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


TURKISH = {"Uniform": "Tekdüze", "Exponential": "Üstel", "Chi-square(5)": "Ki-kare(5)",
           "Lognormal(0.6)": "Lognormal(0,6)", "Pareto(2.5)": "Pareto(2,5)", "Pareto(1.5)": "Pareto(1,5)"}


def name(label):
    """Distribution label in the language of the figure being drawn."""
    from cltspeed.style import L
    return L(label, TURKISH.get(label, label))
