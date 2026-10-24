# Changes

## Version 2.1.0

No result changed: every table and every number in the notes is the same as in version 2.0.0.

- Figures redrawn in the style of the charts on muhammedikbalyilmaz.com: series in black and deep blue (#0033a0),
  red dotted lines for dates and reference values, labels at the end of the lines where possible, bold panel titles,
  Latin Modern Roman. Every figure is 4.75 inches wide, the width of the text in the notes.
- Every figure is drawn in English and in Turkish (decimal comma), as before.
- Axis rule: every value axis starts and ends at a labelled tick, and there are no minor ticks. Before a figure is
  saved, `cltspeed/style.py` checks it and stops with an error if an axis limit is not a labelled tick, if two pieces of
  text overlap, if a note or a legend lies on a data line, or if a note leaves its panel.
- The race animation (`cltspeed/race.py`, `figures/race_exponential.gif`) uses the same colours and axes; the
  command in the README writes it at 150 dpi.
- The project notes now use the note template: 11 pt, A4, 4.5 cm side margins, a header with the short name and
  the compilation date, and the page count. Wide equations and tables were split or set in a smaller size to fit the
  narrower text block.
- References: one paragraph per entry with a 1.27 cm hanging indent and an ampersand before the last author
  (APA 7).
- Compiled Python files (`__pycache__/`) that had been committed are removed from the repository; `.gitignore`
  already excluded them.

## Version 2.0.0

### New

- `cltspeed/` holds the distributions, the exact computations, the simulation and the animated race; the analyses,
  the notebook and the tests use it.
- The exact Kolmogorov distance $D_n$ between the law of the standardized mean and $N(0, 1)$ for Normal, Uniform,
  Laplace, Exponential, Chi-square and Student-t (gamma and Irwin–Hall distribution functions, Gil-Pelaez inversion of
  the characteristic function), and the leading terms of its expansion.
- The noise level of a simulation with $R$ replications (exact distribution of the Kolmogorov statistic).
- Unit tests (`tests/run_tests.py`) against SciPy, numerical Fourier integrals of the densities, exact rational
  arithmetic, closed-form distribution functions and four values printed by version 1.
- Three analyses with fixed seeds, three figures (English and Turkish), CSV tables, and a project note in English and
  Turkish whose numbers are written from the tables.
- The interactive race now shows the exact distance next to the simulated one, the noise level, and the exact
  $F_n - \Phi$; it can also be run without Jupyter (`cltspeed.race.run`).
- `CITATION.cff`, `requirements.txt`, a GitHub Actions workflow that runs the tests.

### Corrections to version 1

| # | Version 1 | Version 2 |
|---|---|---|
| 1 | The race ranked the distributions by a simulated distance with $R = 2000$. Its noise averages about 0.017 even when $Z_n$ is exactly normal: for Normal it reported 0.0151 at $n = 10^4$. For Uniform, Laplace and Student-t(5) the exact distance was below this noise in all 30 frames, for Exponential in 22. | The exact distance is computed and shown next to the simulated one, with the noise level of the simulation (analysis 2). |
| 2 | Laplace was grouped with Exponential as "slower". Laplace is symmetric and its distance falls like $1/n$; at $n = 30$ it is 0.0023 against 0.0243 for Exponential. | Rates and constants come from the exact distance and the Edgeworth expansion (analysis 1). |
| 3 | "Very slow: Student-t (small df)". The default, Student-t(5), reaches $D_n < 0.01$ at $n = 8$. | Student-t(3), with infinite third absolute moment, is added; it needs $n = 116$. |
| 4 | For infinite variance the notebook stopped with a message. | Analysis 3 shows what the sample mean does instead: the Cauchy mean does not concentrate; the Pareto(1.5) mean concentrates like $n^{-1/3}$. |
| 5 | The distance was taken over 401 grid points on $[-12, 12]$. | The exact supremum of the empirical difference, attained at a data point. |
| 6 | The difference panel kept the limits of its empty first state (about $\pm 0.055$); for Pareto(2.5) the curve left the panel in 19 of 30 frames (distance up to 0.151). | Limits follow the data. |
| 7 | Draws outside the histogram window were moved onto its edge bins (`np.clip`). At the default settings this affected at most 0.05% of the draws (Pareto). | Draws outside the window are left out and their share is printed. |
| 8 | A uniform range with $b \le a$ was replaced silently by $b = a + 10^{-12}$. | Invalid parameters raise an error with a message. |
| 9 | A render folder was created at every run, also without recording; the repository held nine of them, nine GIFs, `README.md.rtf`, and HTML and PDF exports of the notebook. The README repeated `pip install pillow`. | Folders are created only when recording. Version 1 kept as `legacy/clt_speed_race_v1.ipynb`; the other files remain in the `v1.0.0` release. |

## Version 1.0.0

First version: an interactive notebook that animates the histogram of the standardized mean and its distance to the
normal distribution for ten distributions, with GIF and MP4 recording.
