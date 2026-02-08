# CLT Speed Race
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive notebook that visualizes **how fast** the **Central Limit Theorem (CLT)** kicks in for different probability distributions.

Instead of only showing *that* convergence happens (when it does), this project compares **convergence speed** by tracking multiple diagnostics as sample size grows.

---

## Mathematical Setup

Let \(X_1,\dots,X_n\) be i.i.d. random variables with mean \(\mu\) and variance \(\sigma^2\) (finite unless noted).

Sample mean:

~~~math
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i
~~~

Standardized mean:

~~~math
Z_n = \frac{\sqrt{n}(\bar X_n-\mu)}{\sigma}
~~~

If \(0 < \sigma^2 < \infty\):

~~~math
Z_n \Rightarrow \mathcal N(0,1)
~~~

The simulation shows **how fast** this convergence occurs.

---

## What the notebook renders

As sample size \(n\) increases, the notebook updates several coordinated views:

- **Histogram of \(Z_n\)**  
  with **standard normal density overlay**
- **ECDF difference** \(F_n(z) - \Phi(z)\)
- **Kolmogorov–Smirnov distance** (KS)
- **Tail mass outside the display window** (clipping report)

This creates a dynamic **convergence speed race** across distributions.

---

## Distributions included

- Normal  
- Uniform  
- Exponential  
- Laplace  
- Lognormal  
- Student-t  
- Chi-square  
- F  
- Pareto  
- Cauchy  

These allow comparison between:

- symmetric vs. skewed  
- light vs. heavy tails  
- finite vs. infinite variance  

---

## Visualization mechanics

To keep plots stable (especially with heavy tails):

- Histogram display is clipped to a **fixed window**
- Tail mass outside the window is **reported**
- KS and ECDF differences use the **unclipped data**

If mean or variance is infinite, CLT visualization stops automatically.

---

## Interactive controls

Users can adjust:

- distribution and parameters  
- sample size growth path  
- Monte Carlo replications  
- random seed  

---

## Recording support (optional)

Each run can optionally export:

- GIF  
- MP4  
- PNG frames  

All frames are saved to a local render folder.

---

## Requirements

- Python ≥ 3.9
- numpy
- matplotlib
- ipywidgets

Install:

~~~bash
pip install numpy matplotlib ipywidgets
~~~

Optional for GIF export:

~~~bash
pip install pillow
~~~

Optional for MP4 export:

~~~bash
brew install ffmpeg
~~~

---

## Interpretation (rule-of-thumb)

Fast convergence:
- Normal  
- Uniform  

Slower:
- Exponential  
- Laplace  

Very slow:
- Student-t (small df)

CLT fails:
- Pareto (\(\alpha \le 2\))  
- Cauchy  

Lack of stabilization in infinite-variance cases is expected.

---

## Author

Muhammed İkbal Yılmaz  
myucanlar@gmail.com
