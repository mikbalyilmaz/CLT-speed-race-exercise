# CLT Speed Race: Exact Distances, Simulation Noise and Heavy Tails

[![tests](https://github.com/mikbalyilmaz/CLT-speed-race-exercise/actions/workflows/tests.yml/badge.svg)](https://github.com/mikbalyilmaz/CLT-speed-race-exercise/actions/workflows/tests.yml)

Version 2 of a small study of how fast the standardized sample mean

$$Z_n = \frac{\sqrt{n}\,(\bar X_n - \mu)}{\sigma}$$

approaches the standard normal distribution for the ten population distributions of version 1. Speed is measured by
the Kolmogorov distance $D_n = \sup_x |P(Z_n \le x) - \Phi(x)|$, computed exactly (without simulation) where possible
and compared with the leading term of its Edgeworth expansion. The animated "race" of version 1 is kept, rebuilt so that
it shows the exact distance and the noise level of the simulation. A short project note (English and Turkish) explains
why each analysis was done and what it shows.

## Contents

| Path | What it is |
|---|---|
| `cltspeed/distributions.py` | The ten distributions of version 1: samplers (identical to version 1), moments, characteristic functions |
| `cltspeed/exact.py` | Exact $D_n$: gamma and Irwin–Hall distribution functions, Gil-Pelaez inversion; leading terms |
| `cltspeed/montecarlo.py` | The simulation of version 1, reproduced draw for draw; exact empirical distance; noise level of a simulation |
| `cltspeed/race.py` | The animated race (frames, GIF, MP4), usable with or without Jupyter |
| `cltspeed/style.py` | Shared figure style (Latin Modern Roman, colours, English and Turkish versions of each figure) |
| `tests/run_tests.py` | Unit tests against SciPy, numerical Fourier integrals, exact rational arithmetic and the output of version 1 |
| `analysis/01_exact_speed.py` | Exact $D_n$ for six distributions and its leading terms (Figure 1, Table 1) |
| `analysis/02_monte_carlo.py` | What the simulation of version 1 measures (Figure 2) |
| `analysis/03_heavy_tails.py` | Lognormal, F and Pareto by simulation; infinite variance (Figure 3) |
| `analysis/run_all.py` | Runs the three analyses in order |
| `notebook/clt_speed_race.ipynb` | The interactive race |
| `figures/` | Figures as PNG (300 dpi) and PDF, every reported table as CSV; `figures/tr/` holds the Turkish figures |
| `paper/clt_note.pdf`, `paper/clt_notu.pdf` | Project note in English and in Turkish; `make_paper.py` writes all their numbers from `figures/*.csv` |
| `legacy/clt_speed_race_v1.ipynb` | Version 1, unchanged |
| `fonts/` | Latin Modern Roman (GUST Font License) |

## How to run

```
pip install -r requirements.txt
python tests/run_tests.py
cd analysis
python run_all.py                    # about four minutes
cd ../paper
python make_paper.py
xelatex clt_note.tex && xelatex clt_note.tex       # English; needs XeLaTeX
xelatex clt_notu.tex && xelatex clt_notu.tex       # Turkish
```

Seeds are fixed, so every number is the same at every run.

For the interactive race: `pip install ipywidgets jupyter`, then open `notebook/clt_speed_race.ipynb`. MP4 recording
needs `ffmpeg`; without it the notebook writes a GIF.

## Usage

```python
from cltspeed import Distribution, kolmogorov_distance, leading_term
from cltspeed import race

d = Distribution("Exponential")
kolmogorov_distance(d, 30)       # (D_30, point where it is attained, method)
leading_term(d, 30)              # (|skewness| phi(0) / (6 sqrt n), order 0.5, "skewness")
rows, gif = race.run(d, R=2000, nmax=10_000, steps=30, seed=42, outdir="renders/exponential", fmt="gif")
```

## Main results

- **Skewness sets the speed.** For symmetric distributions with a finite fourth moment (Uniform, Laplace,
  Student-t(5)) $D_n$ falls like $1/n$; for skewed ones (Chi-square(5), Exponential) only like $1/\sqrt n$, with the
  constant given by the Edgeworth expansion. $D_n < 0.01$ is reached at $n = 3$ (Uniform), 7 (Laplace), 8
  (Student-t(5)), 71 (Chi-square(5)) and 177 (Exponential). Laplace, which version 1 grouped with Exponential, is
  about ten times closer to normal at $n = 30$.
- **Infinite third moments slow convergence.** Student-t(3) is symmetric but converges like $0.110/\sqrt n$ and needs
  $n = 116$. Pareto(2.5), the default of version 1, is still at a distance of 0.031 at $n = 10^4$.
- **The race of version 1 mostly showed simulation noise.** With $R = 2000$ replications the simulated distance averages
  about 0.017–0.019 even when $Z_n$ is exactly normal: version 1 reported 0.0151 for the Normal distribution
  at $n = 10^4$. To resolve the exact distance of Uniform at $n = 1000$ would take about $10^9$ replications.
- **Infinite variance.** The interquartile range of the Cauchy mean stays at 2 for every $n$; for Pareto(1.5) it
  shrinks like $n^{-0.30}$ (theory: $n^{-1/3}$), not like $n^{-1/2}$.

All numbers are in `figures/*.csv` and in the project note.

## Changes from version 1

See [`NEWS.md`](NEWS.md). In short: version 1 ranked the distributions from simulations whose distance was mostly
Monte Carlo noise, grouped Laplace with the skewed Exponential, drew the difference curve with fixed limits that cut off
large values, piled draws outside the histogram window onto its edge bins, replaced an invalid uniform range silently, and
created a render folder at every run. Its repository held nine such folders, nine GIFs, an RTF copy of the README and
HTML and PDF exports of the notebook.

## Limitations

The Kolmogorov distance measures the largest error in a probability, not the relative error in the tails. Lognormal, F
and Pareto have no exact method here and are studied by simulation. Only the default parameters of version 1 are used.

## References

- Gil-Pelaez, J. (1951). Note on the inversion theorem. *Biometrika*, 38(3–4), 481–482.
- Gnedenko, B. V., Kolmogorov, A. N. (1954). *Limit Distributions for Sums of Independent Random Variables*. Addison-Wesley.
- Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer.
- Katz, M. L. (1963). Note on the Berry–Esseen theorem. *Annals of Mathematical Statistics*, 34(3), 1107–1108.

## Citation

Citation metadata are in [`CITATION.cff`](CITATION.cff); GitHub shows them under "Cite this repository".

## Licence

MIT; see [`LICENSE`](LICENSE). The fonts in `fonts/` are distributed under the GUST Font License.
