"""Writes every number and table of the project notes from ../figures/*.csv, so that the text cannot drift
away from the analysis. Run from inside paper/:  python make_paper.py

Output, once in English (suffix _en, for clt_note.tex) and once in Turkish (suffix _tr, for clt_notu.tex):
numbers_xx.tex (one macro per quoted number) and tab_summary_xx.tex (table body ending with its bottom rule,
because a rule cannot follow \\input directly inside a tabular).
"""

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cltspeed import HE3_MAX, PHI0, t3_constant  # noqa: E402

FIG = ROOT / "figures"
OUT = Path(__file__).resolve().parent


def read(name):
    with open(FIG / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


exact = read("table1_exact_distance")
summary = {r["distribution"]: r for r in read("table2_summary")}
race = read("table3_v1_race")
frames = {r["distribution"]: r for r in read("table4_v1_frames")}
needed = {(r["distribution"], int(r["n"])): float(r["replications_needed"]) for r in read("table5_replications")}
noise = read("table6_noise")[0]
heavy = {(r["quantity"], r["distribution"]): r for r in read("table9_heavy_tails_summary")}

NAMES = {
    "en": {"Normal": "Normal", "Uniform": "Uniform", "Exponential": "Exponential", "Laplace": "Laplace",
           "Lognormal(0.6)": "Lognormal(0.6)", "Chi-square(5)": "Chi-square(5)", "F(5, 10)": "F(5, 10)",
           "Student-t(5)": "Student-t(5)", "Student-t(3)": "Student-t(3)", "Pareto(2.5)": "Pareto(2.5)",
           "Cauchy": "Cauchy"},
    "tr": {"Normal": "Normal", "Uniform": "Tekdüze", "Exponential": "Üstel", "Laplace": "Laplace",
           "Lognormal(0.6)": "Lognormal(0,6)", "Chi-square(5)": "Ki-kare(5)", "F(5, 10)": "F(5, 10)",
           "Student-t(5)": "Student-t(5)", "Student-t(3)": "Student-t(3)", "Pareto(2.5)": "Pareto(2,5)",
           "Cauchy": "Cauchy"},
}
ORDER = ["Normal", "Uniform", "Laplace", "Student-t(5)", "Student-t(3)", "Chi-square(5)", "Exponential",
         "Lognormal(0.6)", "F(5, 10)", "Pareto(2.5)", "Cauchy"]


def build(lang):
    big, dmark = ("{,}", ".") if lang == "en" else (".", "{,}")

    def dec(x, d):
        return f"{float(x):.{d}f}".replace(".", dmark)

    def num(x):
        return f"{round(float(x)):,}".replace(",", big)

    def sci(x, d=1):
        mant, exp = f"{float(x):.{d}e}".split("e")
        return f"{mant.replace('.', dmark)}\\times 10^{{{int(exp)}}}"

    def sig(x, s=3):
        """s significant digits; scientific notation below 0.001."""
        x = float(x)
        if x == 0:
            return "0"
        if abs(x) < 1e-3:
            return f"${sci(x, s - 1)}$"
        return dec(x, max(0, s - 1 - math.floor(math.log10(abs(x)))))

    def dval(dist, n):
        return float(next(r["D_n"] for r in exact if r["distribution"] == dist and int(r["n"]) == n))

    m = {}
    # analysis 1
    m["HeThreeMax"] = dec(HE3_MAX, 4)
    m["PhiZeroSixth"] = dec(PHI0 / 6, 4)
    m["TThreeConst"] = dec(t3_constant(), 4)
    m["ExpConst"] = dec(summary["Exponential"]["leading_constant"], 4)
    m["ExpDThirty"] = sig(summary["Exponential"]["D_30"])
    m["LapDThirty"] = sig(summary["Laplace"]["D_30"])
    m["ExpOverLap"] = dec(float(summary["Exponential"]["D_30"]) / float(summary["Laplace"]["D_30"]), 1)
    m["UniDOne"] = sig(dval("Uniform", 1))
    m["ExpDOne"] = sig(dval("Exponential", 1))
    for key, dist in (("Uni", "Uniform"), ("Lap", "Laplace"), ("TFive", "Student-t(5)"), ("TThree", "Student-t(3)"),
                      ("Chi", "Chi-square(5)"), ("Exp", "Exponential")):
        m[f"{key}Ratio"] = dec(dval(dist, 10000) / float(next(r["leading_term"] for r in exact
                                                               if r["distribution"] == dist and int(r["n"]) == 10000)), 3)
        m[f"{key}NBelow"] = num(summary[dist]["n_below_0.01_exact"])
        m[f"{key}NLead"] = num(summary[dist]["n_below_0.01_leading_term"])
        m[f"{key}DTenK"] = sig(dval(dist, 10000))
    m["LogNLead"] = num(summary["Lognormal(0.6)"]["n_below_0.01_leading_term"])
    m["FNLead"] = num(summary["F(5, 10)"]["n_below_0.01_leading_term"])
    m["TThreeSqrtD"] = dec(dval("Student-t(3)", 10000) * 100, 4)

    # analysis 2
    m["NoiseExact"] = dec(noise["noise_mean_exact"], 4)
    m["NoiseQ"] = dec(noise["noise_q95_exact"], 4)
    m["NoiseAsym"] = dec(noise["noise_mean_asymptotic"], 4)
    m["NoiseGrid"] = dec(noise["noise_mean_v1_grid_simulated"], 4)
    v1 = {(r["distribution"], int(r["n"])): r for r in race}
    m["VoneNormalTenK"] = dec(v1[("Normal", 10000)]["ks_v1"], 4)
    m["VoneUniTenK"] = dec(v1[("Uniform", 10000)]["ks_v1"], 4)
    m["VoneExpTenK"] = dec(v1[("Exponential", 10000)]["ks_v1"], 4)
    m["VoneUniTen"] = dec(v1[("Uniform", 10)]["ks_v1"], 4)
    m["UniDTen"] = sig(dval("Uniform", 10))
    for key, dist in (("Normal", "Normal"), ("Uni", "Uniform"), ("Lap", "Laplace"), ("TFive", "Student-t(5)"),
                      ("Chi", "Chi-square(5)"), ("Exp", "Exponential")):
        m[f"{key}FramesBelow"] = num(frames[dist]["frames_exact_below_noise"])
    m["ExpNoiseN"] = num(summary["Exponential"]["n_below_noise_R2000_exact"])
    m["UniNoiseN"] = num(summary["Uniform"]["n_below_noise_R2000_exact"])
    m["ParetoFramesCut"] = num(frames["Pareto(2.5)"]["frames_ks_above_0.055"])
    m["ParetoMaxKs"] = dec(frames["Pareto(2.5)"]["max_ks_v1"], 3)
    m["FFramesCut"] = num(frames["F(5, 10)"]["frames_ks_above_0.055"])
    m["RUniThirty"] = sci(needed[("Uniform", 30)])
    m["RUniThousand"] = sci(needed[("Uniform", 1000)])
    m["RExpThirty"] = num(needed[("Exponential", 30)])
    m["RExpThousand"] = num(needed[("Exponential", 1000)])
    m["RTThreeThousand"] = num(needed[("Student-t(3)", 1000)])

    # analysis 3
    m["IqrSlopeParetoTwoFive"] = dec(heavy[("IQR", "Pareto(2.5)")]["slope"], 2)
    m["IqrSlopeParetoOneFive"] = dec(heavy[("IQR", "Pareto(1.5)")]["slope"], 2)
    m["IqrSlopeCauchy"] = dec(heavy[("IQR", "Cauchy")]["slope"], 3)
    m["CauchyIqrOne"] = dec(heavy[("IQR", "Cauchy")]["value_at_1"], 2)
    m["CauchyIqrTenK"] = dec(heavy[("IQR", "Cauchy")]["value_at_10000"], 2)
    p = heavy[("simulated distance", "Pareto(2.5)")]
    m["ParetoSlope"] = dec(p["slope"], 2)
    m["ParetoTen"] = dec(p["simulated_at_10"], 3)
    m["ParetoHundred"] = dec(p["simulated_at_100"], 3)
    m["ParetoThousand"] = dec(p["simulated_at_1000"], 3)
    m["ParetoTenK"] = dec(p["value_at_10000"], 3)
    lg = heavy[("simulated distance", "Lognormal(0.6)")]
    f = heavy[("simulated distance", "F(5, 10)")]
    m["LogSlope"] = dec(lg["slope"], 2)
    m["FSlope"] = dec(f["slope"], 2)
    m["LogSimTen"] = dec(lg["simulated_at_10"], 3)
    m["LogLeadTen"] = dec(lg["leading_at_10"], 3)
    m["FSimTen"] = dec(f["simulated_at_10"], 3)
    m["FLeadTen"] = dec(f["leading_at_10"], 3)
    m["NoiseHundredK"] = dec(heavy[("noise level", "R = 100000")]["value_at_1"], 4)

    (OUT / f"numbers_{lang}.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8")

    dash = "--"
    lines = []
    for dist in ORDER:
        r = summary[dist]
        if dist == "Cauchy":
            no_mean = "no mean or variance" if lang == "en" else "ortalama ve varyans yok"
            lines.append(f"{NAMES[lang][dist]} & \\multicolumn{{7}}{{l}}{{{no_mean}}} \\\\")
            continue
        g = sig(r["skewness"]) if r["skewness"] != "" else dash
        k = f"${sig(r['excess_kurtosis']).strip('$')}$" if r["excess_kurtosis"] != "" else dash
        if r["leading_constant"] == "":
            lead = "0" if dist == "Normal" else dash
        else:
            power = "n^{-1/2}" if float(r["leading_order"]) == 0.5 else "n^{-1}"
            lead = f"${dec(r['leading_constant'], 4)}\\,{power}$"
        d30 = sig(r["D_30"]) if r["D_30"] != "" else dash
        d1000 = sig(r["D_1000"]) if r["D_1000"] != "" else dash
        nex = num(r["n_below_0.01_exact"]) if r["n_below_0.01_exact"] != "" else dash
        nld = num(r["n_below_0.01_leading_term"]) if r["n_below_0.01_leading_term"] != "" else dash
        lines.append(f"{NAMES[lang][dist]} & {g} & {k} & {lead} & {d30} & {d1000} & {nex} & {nld} \\\\")
    lines.append("\\bottomrule")
    (OUT / f"tab_summary_{lang}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


for lang in ("en", "tr"):
    build(lang)
print("Wrote numbers_*.tex and tab_summary_*.tex (en, tr).")
