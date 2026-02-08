# CLT Speed Race
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive simulation visualizing the **speed of convergence in the Central Limit Theorem (CLT)** across different probability distributions.

The notebook illustrates how quickly the standardized sample mean approaches the standard normal distribution as sample size increases.

---

## Mathematical Setup

Let  
X₁,…,Xₙ be i.i.d. with mean μ and variance σ² (finite unless noted).

Sample mean:
X̄ₙ = (1/n) Σ Xᵢ

Standardized mean:
Zₙ = √n (X̄ₙ − μ) / σ

If 0 < σ² < ∞:
Zₙ → N(0,1)

The simulation shows how fast this convergence occurs.

---

## What the notebook renders

For increasing sample sizes n:

• Histogram of Zₙ  
• Standard normal density overlay  
• ECDF − Normal CDF difference  
• Kolmogorov–Smirnov distance  
• Tail mass outside display window  

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

• symmetric vs skewed  
• light vs heavy tails  
• finite vs infinite variance  

---

## Visualization mechanics

To keep plots stable:

• Histogram display is clipped to a fixed window  
• Tail mass outside the window is reported  
• KS and ECDF differences use the **unclipped data**  

If mean or variance is infinite, CLT visualization stops automatically.

---

## Interactive controls

Users can adjust:

• distribution and parameters  
• sample size growth path  
• Monte Carlo replications  
• random seed  

---

## Recording support

Each run can optionally export:

• GIF  
• MP4  
• PNG frames  

All frames are saved to a local render folder.

---

## Requirements

Python ≥ 3.9

numpy  
matplotlib  
ipywidgets  

Install:
pip install numpy matplotlib ipywidgets

Optional for video export:
pip install pillow  
brew install ffmpeg

---

## Interpretation

Fast convergence:
Normal  
Uniform  

Slower:
Exponential  
Laplace  

Very slow:
Student-t (small df)

CLT fails:
Pareto (α ≤ 2)  
Cauchy  

Lack of stabilization in infinite-variance cases is expected.

---

## Author

Muhammed İkbal Yılmaz  
myucanlar@gmail.com
