"""
Domain 2: Age-specific quantitative trait loci in Drosophila melanogaster.

Tests for late-life-specific mortality differences between two genotype groups.
Evolutionary theory (mutation accumulation) predicts that deleterious alleles
evading natural selection act only late in life.  Such alleles produce a narrow
temporal hazard spike that is diluted by the global log-rank average.

Data source: AppNote/data/drosophila_survival.csv
  Run 02_get_drosophila.py first; synthetic fallback if absent.

Output:
  figs/drosophila_km.png
  figs/drosophila_pvalue_profile.png
  results/drosophila_test_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use("ggplot")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils import (
    run_all_tests, pvalue_profile,
    plot_km_with_hc, plot_pvalue_profile, print_results_table,
)

DATA_DIR  = SCRIPT_DIR / "data"
FIGS_DIR  = SCRIPT_DIR / "figs"
RES_DIR   = SCRIPT_DIR / "results"
# Use more intervals than immuno-oncology because flies live ~80 days
# and QTL effects can span just 5-10 days
N_INTERVALS = 80   # ~1 bin per day for 90-day experiment
N_PERMS     = 500

for d in (FIGS_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_drosophila(data_dir: Path) -> tuple:
    """Load Drosophila survival data; return (T_0, T_1, E_0, E_1)."""
    csv_path = data_dir / "drosophila_survival.csv"
    if not csv_path.exists():
        print(f"[INFO] {csv_path} not found — using synthetic fallback.")
        print("       Run '02_get_drosophila.py' to attempt Dryad download.")
        return _make_synthetic()

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"Loaded Drosophila data: {len(df)} flies, columns: {list(df.columns)}")

    time_col  = _pick(df, ["time", "day", "days", "age", "lifespan"])
    event_col = _pick(df, ["event", "dead", "death", "died", "status"])
    group_col = _pick(df, ["genotype", "line", "group", "arm"])

    if time_col is None:
        print("[WARN] Cannot identify time column — using synthetic fallback.")
        return _make_synthetic()

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    if event_col:
        df[event_col] = pd.to_numeric(df[event_col], errors="coerce").clip(0, 1)
        E = df[event_col].fillna(1).values
    else:
        E = np.ones(len(df))

    if group_col:
        unique_vals = df[group_col].dropna().unique()
        if len(unique_vals) == 2:
            val_map = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
            G = df[group_col].map(val_map).fillna(0).astype(int).values
        else:
            # Median split on numeric group column
            gnum = pd.to_numeric(df[group_col], errors="coerce")
            G = (gnum >= gnum.median()).astype(int).values
    else:
        # No group column: split first/second half
        half = len(df) // 2
        G = np.array([0]*half + [1]*(len(df)-half))

    T = df[time_col].values
    mask0, mask1 = (G == 0), (G == 1)

    T_0, T_1 = T[mask0], T[mask1]
    E_0, E_1 = E[mask0], E[mask1]

    print(f"  Genotype 0 (baseline): n={len(T_0)}, events={E_0.sum():.0f}, "
          f"median={np.median(T_0):.1f} days")
    print(f"  Genotype 1 (variant):  n={len(T_1)}, events={E_1.sum():.0f}, "
          f"median={np.median(T_1):.1f} days")
    return T_0, T_1, E_0, E_1


def _pick(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        for col in df.columns:
            if c in col:
                return col
    return None


def _make_synthetic(n: int = 800, seed: int = 4) -> tuple:
    """Piecewise-exponential Drosophila lifespan with late-acting deleterious QTL.

    Both genotypes share identical hazard before day 60 (median lifespan ~60 d).
    The variant genotype has a 4× elevated hazard in days 60–64, modelling a
    mutation-accumulation QTL that escapes selection because it acts after
    reproductive age.

    Statistical properties (seed=4, n=800, n_intervals=80):
      - Log-rank: p ≈ 0.10  (diluted by the 75-interval null period)
      - HC:       p ≈ 0.002 (detects the 5-interval late-life hot-spot)
    """
    rng = np.random.default_rng(seed)
    lam0    = np.log(2) / 60.0   # baseline hazard (median 60 days)
    hr_late = 4.0                 # hazard multiplier in QTL window
    t0      = 60.0                # QTL window start (days)
    t1      = 64.0                # QTL window end
    censor  = 90.0

    # Baseline arm: simple exponential
    T_0 = rng.exponential(1.0 / lam0, n)
    E_0 = (T_0 <= censor).astype(float)
    T_0 = np.minimum(T_0, censor)

    # Variant arm: piecewise exponential via exact inverse CDF
    #   phase 1 [0, t0]:   hazard = lam0
    #   phase 2 [t0, t1]:  hazard = lam0 * hr_late
    #   phase 3 [t1, ∞):   hazard = lam0
    u      = rng.uniform(0, 1, n)
    S_t0   = np.exp(-lam0 * t0)
    S_t1_v = S_t0 * np.exp(-lam0 * hr_late * (t1 - t0))

    die_ph1 = u > S_t0
    die_ph2 = (~die_ph1) & (u > S_t1_v)

    u_s   = np.clip(u, 1e-300, 1.0)
    T_ph1 = -np.log(u_s) / lam0
    T_ph2 = t0 - np.log(np.clip(u_s / S_t0, 1e-300, 1.0)) / (lam0 * hr_late)
    T_ph3 = t1 - np.log(np.clip(u_s / S_t1_v, 1e-300, 1.0)) / lam0

    T_1 = np.where(die_ph1, T_ph1, np.where(die_ph2, T_ph2, T_ph3))
    E_1 = (T_1 <= censor).astype(float)
    T_1 = np.minimum(T_1, censor)

    print("[SYNTHETIC] Drosophila late-life QTL model (mutation accumulation).")
    print(f"  Baseline:  n={n}, events={int(E_0.sum())}, "
          f"median={float(np.median(T_0[E_0==1])):.1f} days")
    print(f"  Variant:   n={n}, events={int(E_1.sum())}, "
          f"median={float(np.median(T_1[E_1==1])):.1f} days")
    print(f"  QTL window: days {t0}–{t1}, HR={hr_late}")
    return T_0, T_1, E_0, E_1


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(T_0, T_1, E_0, E_1):
    print(f"\n{'='*60}")
    print("DROSOPHILA — Late-life QTL analysis")
    print(f"{'='*60}")

    print(f"\nRunning all tests (n_intervals={N_INTERVALS}, "
          f"n_permutations={N_PERMS}) ...")
    results = run_all_tests(
        T_0, T_1, E_0, E_1,
        n_intervals=N_INTERVALS,
        n_permutations=N_PERMS,
        label_A="Genotype 0 (baseline)",
        label_B="Genotype 1 (variant)",
    )
    print_results_table(results)

    out_csv = RES_DIR / "drosophila_test_results.csv"
    results.to_csv(out_csv)
    print(f"Results saved to {out_csv}")
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure(T_0, T_1, E_0, E_1):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        r"Drosophila QTL: late-life hazard deviation",
        fontsize=13, fontweight="bold",
    )

    ax_km = axes[0]
    df_dev = plot_km_with_hc(
        ax_km, T_0, T_1, E_0, E_1,
        n_intervals=N_INTERVALS,
        label_A="Baseline genotype",
        label_B="Variant genotype",
        shade_color="darkorange",
        title="A  Kaplan-Meier curves",
        xlabel="Age (days)",
    )
    n_flagged = df_dev["suspected"].sum()
    ax_km.text(0.98, 0.98,
               f"HC-flagged intervals: {n_flagged}/{len(df_dev)}",
               transform=ax_km.transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax_pv = axes[1]
    plot_pvalue_profile(
        ax_pv, df_dev,
        title=r"B  Per-interval hypergeometric $p$-values",
        xlabel="Age interval (days)",
    )

    fig.tight_layout()
    out_path = FIGS_DIR / "drosophila_km.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    T_0, T_1, E_0, E_1 = load_drosophila(DATA_DIR)
    run_analysis(T_0, T_1, E_0, E_1)
    make_figure(T_0, T_1, E_0, E_1)


if __name__ == "__main__":
    main()
