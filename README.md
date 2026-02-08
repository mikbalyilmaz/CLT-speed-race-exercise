# CLT Speed Race
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive notebook that visualizes **how fast** the **Central Limit Theorem (CLT)** kicks in for different probability distributions.

---

## Mathematical Setup

Let $X_1,\dots,X_n$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$ (finite unless noted).

Sample mean:

$$
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i
$$

Standardized mean:

$$
Z_n = \frac{\sqrt{n}(\bar X_n-\mu)}{\sigma}
$$

If $0 < \sigma^2 < \infty$:

$$
Z_n \Rightarrow \mathcal N(0,1)
$$

The simulation shows **how fast** this convergence occurs.

---

## What the notebook renders

As sample size $n$ increases:

- Histogram of $Z_n$  
- Standard normal density overlay  
- ECDF difference $F_n - \Phi$  
- Kolmogorov–Smirnov distance  
- Tail mass outside display window  

This produces a dynamic **convergence speed race** across distributions.

---

## Distributions included

Normal  
Uniform  
Exponential  
Laplace  
Lognormal  
Student-t  
Chi-square  
F  
Pareto  
Cauchy  

These allow comparison between:

- symmetric vs skewed  
- light vs heavy tails  
- finite vs infinite variance  

---

## Visualization mechanics

To keep plots stable:

- Histogram display is clipped to a fixed window  
- Tail mass outside the window is reported  
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

Each run can export:

- GIF  
- MP4  
- PNG frames  

All frames are saved to a local render folder.

---

## Requirements

Python ≥ 3.9
pip install numpy matplotlib ipywidgets

Optional for GIF:

pip install pillow

pip install pillow


Optional for MP4:



brew install ffmpeg


---

## Interpretation

Fast convergence:
- Normal  
- Uniform  

Slower:
- Exponential  
- Laplace  

Very slow:
- Student-t (small df)

CLT fails:
- Pareto ($\alpha \le 2$)  
- Cauchy  

Lack of stabilization in infinite-variance cases is expected.

---

## Author

Muhammed İkbal Yılmaz  
myucanlar@gmail.com
