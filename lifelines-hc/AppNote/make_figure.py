"""
Generate the composite Application Note figure (Figure 1).

Layout (2 rows × 2 columns):
  Row 1 — Immuno-oncology (POPLAR trial)
    Panel A: KM curves with HC-flagged intervals (gray shading)
    Panel B: Per-interval -log10(p) bar chart with HC threshold line
  Row 2 — Transcriptomics (DDX5, SCAN-B breast cancer)
    Panel C: KM curves with HC-flagged intervals
    Panel D: Per-interval -log10(p) bar chart with HC threshold line

Each KM panel shows:
  - Kaplan-Meier survival curves for the two groups
  - Gray-shaded vertical bands for HC Delta* intervals
  - Legend with HC statistic and p-value

Each profile panel shows:
  - Bar chart: -log10(hypergeometric p-value) at each time bin
  - Dashed line: HC p-value threshold
  - Bars above threshold are highlighted in red

Usage:
    python make_figure.py [--out figs/figure1.png]

Pre-requisites:
  run_immuno_oncology.py  (or POPLAR.csv in data/)
  run_scanb.py            (or SCANB dataset on path)
"""

import argparse
import os
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

N_INTERVALS_IO    = 60    # immuno-oncology  (~1 bin/month for 60-mo trial)
N_INTERVALS_SCANB = 100   # transcriptomics  (days, as in original paper)
N_PERMS           = 500
GENE              = "DDX5"   # primary showcase gene

# Paths that run_immuno_oncology.py and run_scanb.py understand
_DEFAULT_SCANB_PATHS = [
    Path(os.environ.get("SCANB_PATH", "")),
    DATA_DIR / "SCANB_groups_valid.csv",
    DATA_DIR / "SCANB_groups_valid_KS_censored.csv",
    SCRIPT_DIR.parent.parent / "Data" / "SCANB_groups_valid.csv",
    SCRIPT_DIR.parent.parent / "Data" / "SCANB_groups_valid_KS_censored.csv",
    Path("/Users/kipnisal/Dropbox/Research/survival/Data/SCANB_groups_valid.csv"),
    Path("/Users/kipnisal/Dropbox/Research/survival/Data/SCANB_groups_valid_KS_censored.csv"),
]


# ---------------------------------------------------------------------------
# Data loaders (re-implemented here to keep make_figure.py self-contained)
# ---------------------------------------------------------------------------

def load_poplar():
    """Load CheckMate 057 PFS trial data or generate synthetic fallback.

    Prefers Checkmate057_1C.csv (Borghaei 2015 NEJM, PFS, n=582).
    Falls back to POPLAR.csv, then synthetic crossing-curve model.
    """
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


def load_scanb_gene(gene: str):
    """Load SCANB data for a specific gene."""
    for p in _DEFAULT_SCANB_PATHS:
        if p and str(p) and p.exists() and p.is_file():
            df = pd.read_csv(p, usecols=["time", "event", gene])
            df = df.dropna()
            df["time"]  = pd.to_numeric(df["time"],  errors="coerce")
            df["event"] = pd.to_numeric(df["event"], errors="coerce")
            df = df.dropna(subset=["time"])
            g0 = df[df[gene]==0]; g1 = df[df[gene]==1]
            return (g0["time"].values, g1["time"].values,
                    g0["event"].values, g1["event"].values)
    raise FileNotFoundError(
        f"SCANB data not found.  Run run_scanb.py or set SCANB_PATH."
    )


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
    # --- Load data ---
    T_ctrl, T_trt, E_ctrl, E_trt = load_poplar()
    T_0, T_1, E_0, E_1 = load_scanb_gene(GENE)

    # --- Compute deviations (needed for both panels in each row) ---
    dev_io = pvalue_profile(T_ctrl, T_trt, E_ctrl, E_trt,
                             n_intervals=N_INTERVALS_IO)
    dev_sc = pvalue_profile(T_0, T_1, E_0, E_1,
                             n_intervals=N_INTERVALS_SCANB)

    # --- Run HC tests for statistics to display ---
    res_io = run_all_tests(T_ctrl, T_trt, E_ctrl, E_trt,
                            n_intervals=N_INTERVALS_IO,
                            n_permutations=N_PERMS)
    res_sc = run_all_tests(T_0, T_1, E_0, E_1,
                            n_intervals=N_INTERVALS_SCANB,
                            n_permutations=N_PERMS)

    # --- Figure layout ---
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # --- Panel A: Immuno-oncology KM ---
    plot_km_with_hc(
        ax_A, T_ctrl, T_trt, E_ctrl, E_trt,
        n_intervals=N_INTERVALS_IO,
        label_A="Docetaxel (control)",
        label_B="Nivolumab",
        shade_color="steelblue",
        title="A  Immuno-oncology (CheckMate 057 PFS)",
        xlabel="Time (months)",
    )
    _annotate_pvals(ax_A, res_io)

    # --- Panel B: Immuno-oncology p-value profile ---
    plot_pvalue_profile(
        ax_B, dev_io,
        title=r"B  Interval $p$-values — CheckMate 057",
        xlabel="Time interval (months)",
    )

    # --- Panel C: Transcriptomics KM ---
    plot_km_with_hc(
        ax_C, T_0, T_1, E_0, E_1,
        n_intervals=N_INTERVALS_SCANB,
        label_A=f"{GENE}: low expression",
        label_B=f"{GENE}: high expression",
        shade_color="mediumorchid",
        title=f"C  Transcriptomics (SCAN-B, {GENE})",
        xlabel="Follow-up time (days)",
    )
    _annotate_pvals(ax_C, res_sc)

    # --- Panel D: Transcriptomics p-value profile ---
    plot_pvalue_profile(
        ax_D, dev_sc,
        title=r"D  Interval $p$-values — transcriptomics",
        xlabel="Follow-up interval (days)",
    )

    # --- Save ---
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nFigure 1 saved to {out_path}")
    plt.close(fig)


def _annotate_pvals(ax, results: pd.DataFrame) -> None:
    """Add HC and log-rank p-values as a text box on *ax*."""
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
    global GENE
    parser = argparse.ArgumentParser(description="Generate Application Note Figure 1")
    parser.add_argument("--out", default=str(FIGS_DIR / "figure1.png"),
                        help="Output file path")
    parser.add_argument("--gene", default=GENE,
                        help="SCANB gene to feature in the figure")
    args = parser.parse_args()

    GENE = args.gene
    build_figure(Path(args.out))


if __name__ == "__main__":
    main()
