"""The animated "speed race" of the notebook, written as plain functions so that it can be tested and run
without Jupyter.

Each frame shows, for one sample size n:
(a) the histogram of the R simulated standardized means Z_n on a fixed window; draws outside the window are
    left out and their share is printed (version 1 moved them onto the edge bins);
(b) F_hat(x) - Phi(x), with the exact F_n(x) - Phi(x) as a dashed line when it can be computed;
(c) the race itself: the simulated distance against n, the exact distance D_n, and the noise level of a
    simulation with R replications, below which differences cannot be seen.

Author: Muhammed Ikbal Yilmaz
"""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .exact import difference_function, kolmogorov_distance
from .montecarlo import ks_exact, noise_floor, v1_sample_sizes

WINDOW = 6.0
BINS = np.linspace(-WINDOW, WINDOW, 121)
CURVE_X = np.linspace(-WINDOW, WINDOW, 601)
EXACT_X = np.linspace(-WINDOW, WINDOW, 241)


def simulate(dist, R=2000, nmax=10_000, steps=30, seed=42):
    """Yields (n, z) for the sample sizes of version 1, with the draws of version 1."""
    if not dist.clt_applies():
        raise ValueError(f"{dist.label}: the mean or the variance does not exist, so Z_n is not defined "
                         "(see analysis 3 of the project note)")
    rng = np.random.default_rng(int(seed))
    mu, sd = dist.mean(), dist.sd()
    s = np.zeros(int(R))
    nprev = 0
    for n in v1_sample_sizes(int(nmax), int(steps)):
        s += dist.sample(rng, (int(R), n - nprev)).sum(1)
        nprev = n
        z = math.sqrt(n) * (s / n - mu) / sd
        yield int(n), z[np.isfinite(z)]


def exact_distances(dist, ns):
    """Exact D_n for every n in ns, or None if the distribution has no exact method here."""
    if not (dist.has_cf() or dist.name in ("Exponential", "Chi-square")):
        return None
    return np.array([kolmogorov_distance(dist, int(n))[0] for n in ns])


def make_figure():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    return fig, axes


def draw_frame(fig, axes, dist, n, z, history, exact_ns=None, exact_d=None, R=None, show_exact_curve=True):
    """Draws one frame. `history` is a list of (n, simulated distance) up to and including this n."""
    from .style import BAND, BLUE, INK, MUTED, auto_axis, fit_log_axis, nice_limits, value_axis
    ax_h, ax_d, ax_r = axes
    for ax in axes:
        ax.clear()
    R = R or z.size
    floor_mean, floor_q95 = noise_floor(z.size)

    inside = np.abs(z) <= WINDOW
    outside = 1 - inside.mean()
    heights, _, _ = ax_h.hist(z[inside], bins=BINS, weights=np.full(inside.sum(), 1 / (z.size * (BINS[1] - BINS[0]))),
              color=BAND, edgecolor=BAND, linewidth=0.3)
    xs = np.linspace(-WINDOW, WINDOW, 400)
    ax_h.plot(xs, np.exp(-0.5 * xs * xs) / math.sqrt(2 * math.pi), color=INK, lw=1.4)
    value_axis(ax_h, "x", -WINDOW, WINDOW, 3)
    auto_axis(ax_h, "y", [0.0, max(float(np.max(heights)), 0.4)], intervals=(2, 4))
    ax_h.set_title(f"$Z_n$, $n$ = {n:,}; outside: {100 * outside:.2f}%")
    ax_h.set_xlabel("$z$")

    zs = np.sort(z)
    diff = np.searchsorted(zs, CURVE_X, side="right") / zs.size - 0.5 * (1 + np.vectorize(math.erf)(CURVE_X / math.sqrt(2)))
    ax_d.plot(CURVE_X, diff, color=BLUE, lw=1.0, label="Simulated")
    top = max(0.02, 1.15 * float(np.max(np.abs(diff))))
    if show_exact_curve and exact_d is not None:
        f, _ = difference_function(dist, n)
        exact_diff = f(EXACT_X)
        ax_d.plot(EXACT_X, exact_diff, color=INK, lw=1.2, ls=(0, (3, 2)), label="Exact")
        top = max(top, 1.15 * float(np.max(np.abs(exact_diff))))
    ax_d.axhline(0, color=MUTED, lw=0.6)
    lo_, hi_, step_ = nice_limits(-top, top, (2, 4))
    value_axis(ax_d, "y", -max(-lo_, hi_), max(-lo_, hi_), step_)
    value_axis(ax_d, "x", -WINDOW, WINDOW, 3)
    ax_d.set_title(r"$\hat F(x) - \Phi(x)$")
    ax_d.set_xlabel("$x$")
    ax_d.legend(loc="upper right", fontsize=6.5)

    hn, hd = zip(*history)
    ax_r.plot(hn, hd, "o-", color=BLUE, ms=2.5, lw=1, label="Simulated distance")
    if exact_d is not None:
        ax_r.plot(exact_ns, exact_d, color=INK, lw=1.4, label="Exact $D_n$")
    ax_r.axhspan(1e-12, floor_mean, color=MUTED, alpha=0.15, lw=0)
    ax_r.axhline(floor_mean, color=MUTED, lw=0.9, ls=(0, (1, 1.4)), label=f"Noise level, R = {R:,}")
    lo = min([floor_mean / 3] + ([float(np.min(exact_d)) / 2] if exact_d is not None else []))
    fit_log_axis(ax_r, "y", [lo, 1.0])
    fit_log_axis(ax_r, "x", [exact_ns[0] if exact_ns is not None else hn[0],
                             exact_ns[-1] if exact_ns is not None else max(hn)])
    ax_r.set_title("Distance to the normal distribution")
    ax_r.set_xlabel("$n$")
    ax_r.legend(loc="lower left", fontsize=6.5)
    fig.suptitle(f"{dist.label}: simulated distance {history[-1][1]:.4f}"
                 + (f", exact $D_n$ = {exact_d[list(exact_ns).index(n)]:.3g}" if exact_d is not None else ""),
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold")


def frames_to_gif(frames_dir, gif_path, fps=10):
    from PIL import Image
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        raise RuntimeError("no frames to write")
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frames]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000 / max(1, fps)), loop=0)


def frames_to_mp4(frames_dir, mp4_path, fps=10):
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(max(1, fps)),
                    "-i", str(Path(frames_dir) / "frame_%05d.png"), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(mp4_path)], check=True)


def run(dist, R=2000, nmax=10_000, steps=30, seed=42, outdir=None, fmt="gif", fps=10, dpi=110, on_frame=None):
    """Runs the race. If `outdir` is given, frames are saved there and turned into a GIF or an MP4 (fmt =
    "gif", "mp4" or "frames"); `on_frame(fig)` is called after every frame (the notebook uses it to display
    the figure). Returns (rows, path of the recording or None)."""
    import matplotlib.pyplot as plt
    from .style import use_style
    use_style()
    ns = v1_sample_sizes(int(nmax), int(steps))
    exact_d = exact_distances(dist, ns)
    fig, axes = make_figure()
    frames_dir = None
    if outdir is not None:
        frames_dir = Path(outdir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    history, rows = [], []
    for i, (n, z) in enumerate(simulate(dist, R, nmax, steps, seed)):
        d = ks_exact(z)
        history.append((n, d))
        rows.append({"n": n, "simulated": d, "exact": None if exact_d is None else float(exact_d[i]),
                     "outside_window": float(np.mean(np.abs(z) > WINDOW))})
        draw_frame(fig, axes, dist, n, z, history, ns, exact_d, int(R))
        if frames_dir is not None:
            fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=dpi)
        if on_frame is not None:
            on_frame(fig)
    plt.close(fig)
    path = None
    if frames_dir is not None and fmt != "frames":
        stem = Path(outdir) / f"race_{dist.name}"
        if fmt == "mp4":
            try:
                frames_to_mp4(frames_dir, stem.with_suffix(".mp4"), fps)
                path = stem.with_suffix(".mp4")
            except (RuntimeError, subprocess.CalledProcessError):
                fmt = "gif"  # no ffmpeg: fall back to a GIF
        if fmt == "gif":
            frames_to_gif(frames_dir, stem.with_suffix(".gif"), fps)
            path = stem.with_suffix(".gif")
    return rows, path
