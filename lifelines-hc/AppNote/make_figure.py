"""
Generate the composite Application Note figure (Figure 1).

Layout (2 rows × 2 columns):
  Row 1 — Immuno-oncology (CheckMate 057 PFS)
    Panel A: KM curves with HC-flagged intervals
    Panel B: Per-interval -log10(p) bar chart with HC threshold line
  Row 2 — Adjuvant bisphosphonate therapy (AZURE trial DFS)
    Panel C: KM curves with HC-flagged intervals
    Panel D: Per-interval -log10(p) bar chart with HC threshold line

Usage:
    python make_figure.py [--out figs/figure1.png]

Pre-requisites:
  run_immuno_oncology.py  (or Checkmate057_1C.csv in data/)
  run_azure.py            (or AZURE_2A.csv in data/)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.style.use("ggplot")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import (
    run_all_tests, pvalue_profile,
    plot_km_with_hc, plot_pvalue_profile,
)

DATA_DIR = SCRIPT_DIR / "data"
FIGS_DIR = SCRIPT_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

N_INTERVALS_IO    = 60   # immuno-oncology (~1 bin/month for 30-mo PFS)
N_INTERVALS_AZURE = 80   # bisphosphonate  (~1.5-month bins, 120-mo follow-up)
N_PERMS           = 500


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_poplar():
    """Load CheckMate 057 PFS data or synthetic fallback."""
    for fname in ["Checkmate057_1C.csv", "POPLAR.csv"]:
        csv_path = DATA_DIR / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()
            time_col  = _pick(df, ["time", "os", "os_time", "t"])
            event_col = _pick(df, ["status", "event", "os_event", "dead"])
            arm_col   = _pick(df, ["arm", "trt", "treatment", "group"])
            if time_col and event_col:
                df[time_col]  = pd.to_numeric(df[time_col],  errors="coerce")
                df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
                df = df.dropna(subset=[time_col, event_col])
                if arm_col:
                    vals = sorted(df[arm_col].unique())
                    df["_arm"] = df[arm_col].map({vals[0]: 0, vals[1]: 1}).fillna(0)
                else:
                    df["_arm"] = [0]*(len(df)//2) + [1]*(len(df)-len(df)//2)
                ctrl = df[df["_arm"]==0]; trt = df[df["_arm"]==1]
                return (ctrl[time_col].values, trt[time_col].values,
                        ctrl[event_col].values, trt[event_col].values)

    print("[INFO] Trial CSV not found; using synthetic crossover-curve fallback.")
    rng = np.random.default_rng(9)
    n, lam0, hr_early, hr_late, t0, censor = 200, np.log(2)/10, 1.60, 0.65, 3.0, 24.0
    T_ctrl = rng.exponential(1/lam0, n)
    E_ctrl = (T_ctrl <= censor).astype(float); T_ctrl = np.minimum(T_ctrl, censor)
    S0 = np.exp(-lam0 * hr_early * t0)
    u = rng.uniform(0, 1, n)
    die_early = u > S0
    T_ph1 = -np.log(u) / (lam0 * hr_early)
    T_ph2 = t0 - np.log(np.clip(u / S0, 1e-300, 1.0)) / (lam0 * hr_late)
    T_trt = np.where(die_early, T_ph1, T_ph2)
    E_trt = (T_trt <= censor).astype(float); T_trt = np.minimum(T_trt, censor)
    return T_ctrl, T_trt, E_ctrl, E_trt


def load_azure():
    """Load AZURE trial DFS data or synthetic delayed-benefit fallback."""
    csv_path = DATA_DIR / "AZURE_2A.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        ctrl = df[df["arm"] == "control"]
        trt  = df[df["arm"] == "zoledronic_acid"]
        return (ctrl["time"].values.astype(float), trt["time"].values.astype(float),
                ctrl["event"].values.astype(float), trt["event"].values.astype(float))

    print("[INFO] AZURE_2A.csv not found; using synthetic delayed-benefit fallback.")
    rng  = np.random.default_rng(7)
    n, lam0, t0, hr, censor = 1200, np.log(2)/80, 24.0, 0.75, 120.0
    T_ctrl = rng.exponential(1/lam0, n)
    E_ctrl = (T_ctrl <= censor).astype(float); T_ctrl = np.minimum(T_ctrl, censor)
    S_t0 = np.exp(-lam0 * t0); u = rng.uniform(0, 1, n)
    die_early = u > S_t0
    T_trt = np.where(die_early, -np.log(np.clip(u, 1e-300, 1))/lam0,
                     t0 - np.log(np.clip(u/S_t0, 1e-300, 1))/(lam0*hr))
    E_trt = (T_trt <= censor).astype(float); T_trt = np.minimum(T_trt, censor)
    return T_ctrl, T_trt, E_ctrl, E_trt


def _pick(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        for col in df.columns:
            if c in col: return col
    return None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(out_path: Path):
    T_io_ctrl, T_io_trt, E_io_ctrl, E_io_trt = load_poplar()
    T_az_ctrl, T_az_trt, E_az_ctrl, E_az_trt = load_azure()

    dev_io = pvalue_profile(T_io_ctrl, T_io_trt, E_io_ctrl, E_io_trt,
                             n_intervals=N_INTERVALS_IO)
    dev_az = pvalue_profile(T_az_ctrl, T_az_trt, E_az_ctrl, E_az_trt,
                             n_intervals=N_INTERVALS_AZURE)

    res_io = run_all_tests(T_io_ctrl, T_io_trt, E_io_ctrl, E_io_trt,
                            n_intervals=N_INTERVALS_IO, n_permutations=N_PERMS)
    res_az = run_all_tests(T_az_ctrl, T_az_trt, E_az_ctrl, E_az_trt,
                            n_intervals=N_INTERVALS_AZURE, n_permutations=N_PERMS)

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    plot_km_with_hc(
        ax_A, T_io_ctrl, T_io_trt, E_io_ctrl, E_io_trt,
        n_intervals=N_INTERVALS_IO,
        label_A="Docetaxel (control)",
        label_B="Nivolumab",
        shade_color="steelblue",
        title="A  Immuno-oncology (CheckMate 057 PFS)",
        xlabel="Time (months)",
    )
    _annotate_pvals(ax_A, res_io)

    plot_pvalue_profile(
        ax_B, dev_io,
        title=r"B  Interval $p$-values — CheckMate 057",
        xlabel="Time interval (months)",
    )

    plot_km_with_hc(
        ax_C, T_az_ctrl, T_az_trt, E_az_ctrl, E_az_trt,
        n_intervals=N_INTERVALS_AZURE,
        label_A="Control",
        label_B="Zoledronic acid",
        shade_color="darkorange",
        title="C  Adjuvant bisphosphonate (AZURE trial DFS)",
        xlabel="Time (months)",
    )
    _annotate_pvals(ax_C, res_az)

    plot_pvalue_profile(
        ax_D, dev_az,
        title=r"D  Interval $p$-values — AZURE trial",
        xlabel="Time interval (months)",
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure 1 saved to {out_path}")
    plt.close(fig)


def _annotate_pvals(ax, results: pd.DataFrame) -> None:
    lr_p = results.loc["Log-rank", "p_value"] \
           if "Log-rank" in results.index else float("nan")
    hc_p = results.loc["Higher Criticism (HC)", "p_value"] \
           if "Higher Criticism (HC)" in results.index else float("nan")
    text = f"Log-rank  p = {lr_p:.3f}\nHC         p = {hc_p:.3f}"
    ax.text(0.98, 0.42, text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Application Note Figure 1")
    parser.add_argument("--out", default=str(FIGS_DIR / "figure1.png"),
                        help="Output file path")
    args = parser.parse_args()
    build_figure(Path(args.out))


if __name__ == "__main__":
    main()
