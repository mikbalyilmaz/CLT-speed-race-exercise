"""Shared figure style.

Typeface: Latin Modern Roman, the typeface of LaTeX documents, registered from ../fonts so that no system
installation is needed. Colours: a fixed set of hues; where more than three series share a panel, line styles
and markers distinguish them as well, so that no series is identified by colour alone.

Every figure is drawn twice: in English (figures/*.png and figures/*.pdf, used by the README) and in
Turkish (figures/tr/*.pdf, used by the project note), with the decimal comma in the Turkish version.

Author: Muhammed Ikbal Yilmaz
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FONT_DIR = ROOT / "fonts"

INK, INK2, MUTED, GRID = "#000000", "#333333", "#6f6f6f", "#d9d9d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PURPLE, GOLD, PINK = "#7b4fd0", "#b8860b", "#d0457f"

LANG = "en"


def L(en, tr):
    """English or Turkish label, depending on the figure being drawn."""
    return tr if LANG == "tr" else en


def num(value, fmt="{:g}"):
    """Format a number with the decimal mark of the current figure language."""
    text = fmt.format(value)
    return text.replace(".", ",") if LANG == "tr" else text


def use_style():
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # silences font-subsetting notices
    for path in FONT_DIR.glob("*.otf"):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10, "axes.titlelocation": "left",
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 3,  # Latin Modern is a CFF font; Type 3 embedding keeps every PDF viewer happy
    })


def decimal_axis(ax, axis="y"):
    """Tick labels on a linear axis with the decimal mark of the current language."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(ticker.FuncFormatter(lambda v, _: num(round(v, 10))))


def save_figure(name, draw, heading=None):
    """Draw the figure in both languages. `draw()` returns a figure; `heading()` returns its title, which is
    shown in the PNG only (in the notes the caption carries the title)."""
    global LANG
    (FIG_DIR / "tr").mkdir(parents=True, exist_ok=True)
    for lang in ("en", "tr"):
        LANG = lang
        fig = draw()
        if lang == "en":
            fig.savefig(FIG_DIR / f"{name}.pdf")
            if heading is not None:
                fig.suptitle(heading(), x=0.01, ha="left", fontsize=11)
            fig.savefig(FIG_DIR / f"{name}.png")
        else:
            fig.savefig(FIG_DIR / "tr" / f"{name}.pdf")
        plt.close(fig)
    LANG = "en"
