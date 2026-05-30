"""
Domain 3: Biphasic / opposing hazard trends in breast cancer transcriptomics.

Analyses the SCAN-B breast cancer dataset (Saal et al. 2015) to identify genes
whose expression produces *opposing* hazard trends at different disease stages.
A gene that increases short-term risk (e.g. early metastasis) but reduces
long-term risk (e.g. late chemosensitivity) will show a net null log-rank
p-value because the opposing effects cancel.  HC detects both directional
signals independently.

Focus genes from the Biometrika paper (log-rank fails, HC detects):
  DDX5, ANKLE2, CLCF1, IL10RB

Data source:
  SCANB_groups_valid_KS_censored.csv
  Default search path: ../../Data/SCANB_groups_valid_KS_censored.csv
  Set SCANB_PATH env var to override.

Output:
  figs/scanb_<GENE>_km.png        — per-gene KM + HC figure
  results/scanb_test_results.csv  — comparison table across all focus genes
"""

import os
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

FIGS_DIR = SCRIPT_DIR / "figs"
RES_DIR  = SCRIPT_DIR / "results"
for d in (FIGS_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Candidate search paths for the SCANB data file.
# We prefer SCANB_groups_valid.csv (all genes pass basic validity checks)
# over SCANB_groups_valid_KS_censored.csv (stricter: removes genes where
# censoring pattern differs between groups, which excludes DDX5/ANKLE2/CLCF1).
_DEFAULT_SCANB_PATHS = [
    Path(os.environ.get("SCANB_PATH", "")),
    SCRIPT_DIR / "data" / "SCANB_groups_valid.csv",
    SCRIPT_DIR / "data" / "SCANB_groups_valid_KS_censored.csv",
    SCRIPT_DIR.parent.parent / "Data" / "SCANB_groups_valid.csv",
    SCRIPT_DIR.parent.parent / "Data" / "SCANB_groups_valid_KS_censored.csv",
    Path("/Users/kipnisal/Dropbox/Research/survival/Data/SCANB_groups_valid.csv"),
    Path("/Users/kipnisal/Dropbox/Research/survival/Data/SCANB_groups_valid_KS_censored.csv"),
]

# Focus genes from the Biometrika paper
FOCUS_GENES = ["DDX5", "ANKLE2", "CLCF1", "IL10RB"]
N_INTERVALS = 100   # as used in the original paper
N_PERMS     = 500


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_scanb_path() -> Path | None:
    for p in _DEFAULT_SCANB_PATHS:
        if p and str(p) and p.exists() and p.is_file():
            return p
    return None


def load_scanb(gene: str, scanb_path: Path) -> tuple:
    """Load SCANB data for a specific gene; return (T_low, T_high, E_low, E_high).

    The gene column in SCANB is a binary indicator: 1 = expression ≥ median
    (high group), 0 = expression < median (low group).  Following the original
    paper, the comparison is low (group 0) vs high (group 1) by default.
    For IL10RB the paper uses expression > median as the index group.
    """
    df = pd.read_csv(scanb_path, usecols=["time", "event", gene])
    df = df.dropna()
    df["time"]  = pd.to_numeric(df["time"],  errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df = df.dropna(subset=["time"])

    g0 = df[df[gene] == 0]
    g1 = df[df[gene] == 1]

    T_0 = g0["time"].values;  E_0 = g0["event"].values
    T_1 = g1["time"].values;  E_1 = g1["event"].values

    return T_0, T_1, E_0, E_1


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_gene(gene: str, scanb_path: Path) -> pd.DataFrame:
    """Run all tests for one gene; return comparison DataFrame with gene label."""
    print(f"\n--- {gene} ---")
    T_0, T_1, E_0, E_1 = load_scanb(gene, scanb_path)
    print(f"  Low expr:  n={len(T_0)}, events={E_0.sum():.0f}")
    print(f"  High expr: n={len(T_1)}, events={E_1.sum():.0f}")

    results = run_all_tests(
        T_0, T_1, E_0, E_1,
        n_intervals=N_INTERVALS,
        n_permutations=N_PERMS,
        label_A=f"{gene} low",
        label_B=f"{gene} high",
    )
    results.insert(0, "gene", gene)
    print_results_table(results.drop(columns="gene"))
    return results


def make_gene_figure(gene: str, scanb_path: Path) -> None:
    """Two-panel figure for one gene: KM + p-value profile."""
    T_0, T_1, E_0, E_1 = load_scanb(gene, scanb_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"SCAN-B breast cancer — gene {gene}", fontsize=13,
                 fontweight="bold")

    ax_km = axes[0]
    df_dev = plot_km_with_hc(
        ax_km, T_0, T_1, E_0, E_1,
        n_intervals=N_INTERVALS,
        label_A=f"{gene}: low expression",
        label_B=f"{gene}: high expression",
        shade_color="mediumorchid",
        title="A  Kaplan-Meier curves",
        xlabel="Follow-up time (days)",
    )
    n_flagged = df_dev["suspected"].sum()
    ax_km.text(0.98, 0.98,
               f"HC-flagged: {n_flagged}/{len(df_dev)}",
               transform=ax_km.transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax_pv = axes[1]
    plot_pvalue_profile(
        ax_pv, df_dev,
        title=r"B  Per-interval hypergeometric $p$-values",
        xlabel="Follow-up interval (days)",
    )

    fig.tight_layout()
    out_path = FIGS_DIR / f"scanb_{gene}_km.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close(fig)


def make_summary_figure(all_results: pd.DataFrame) -> None:
    """4-panel summary: one KM+HC plot per gene, arranged in 2×2 grid."""
    scanb_path = find_scanb_path()
    genes = all_results["gene"].unique()
    n = len(genes)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.5 * nrows))
    axes_flat = np.array(axes).flatten()
    fig.suptitle("SCAN-B breast cancer: HC-rescued biomarker genes",
                 fontsize=13, fontweight="bold")

    for i, gene in enumerate(genes):
        ax = axes_flat[i]
        T_0, T_1, E_0, E_1 = load_scanb(gene, scanb_path)
        df_dev = plot_km_with_hc(
            ax, T_0, T_1, E_0, E_1,
            n_intervals=N_INTERVALS,
            label_A=f"{gene} low",
            label_B=f"{gene} high",
            shade_color="mediumorchid",
            title=f"{gene}",
            xlabel="Follow-up (days)",
        )
        # Add HC and log-rank p-values as text annotation
        row = all_results[all_results["gene"] == gene]
        if not row.empty:
            lr_p  = row.loc[row.index[row["gene"]==gene][0], "p_value"] \
                    if False else None  # handled below
            lr_p  = all_results.loc[(all_results["gene"]==gene) &
                                     (all_results.index=="Log-rank"), "p_value"]
            hc_p  = all_results.loc[(all_results["gene"]==gene) &
                                     (all_results.index=="Higher Criticism (HC)"), "p_value"]
            if not lr_p.empty and not hc_p.empty:
                ax.text(
                    0.98, 0.40,
                    f"LR  p={lr_p.values[0]:.3f}\n"
                    f"HC  p={hc_p.values[0]:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85),
                )

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out_path = FIGS_DIR / "scanb_summary.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\nSummary figure saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scanb_path = find_scanb_path()
    if scanb_path is None:
        print("[ERROR] SCANB data file not found.  Searched paths:")
        for p in _DEFAULT_SCANB_PATHS:
            print(f"  {p}")
        print("\nSet the SCANB_PATH environment variable or place the file at:")
        print("  AppNote/data/SCANB_groups_valid_KS_censored.csv")
        sys.exit(1)

    print(f"SCANB data: {scanb_path}  ({sum(1 for _ in open(scanb_path))-1} patients)")
    print(f"\n{'='*60}")
    print("SCAN-B BREAST CANCER — Focus gene analysis")
    print(f"{'='*60}")

    # Check which focus genes are available
    header_cols = pd.read_csv(scanb_path, nrows=0).columns.tolist()
    available = [g for g in FOCUS_GENES if g in header_cols]
    missing   = [g for g in FOCUS_GENES if g not in header_cols]
    if missing:
        print(f"[WARN] Genes not found in dataset: {missing}")
    if not available:
        print("[ERROR] No focus genes found in SCANB file.  Exiting.")
        sys.exit(1)
    print(f"Analysing genes: {available}")

    # Run analysis for each gene
    all_results = []
    for gene in available:
        results = run_gene(gene, scanb_path)
        all_results.append(results)
        make_gene_figure(gene, scanb_path)

    combined = pd.concat(all_results, ignore_index=False)
    out_csv = RES_DIR / "scanb_test_results.csv"
    combined.to_csv(out_csv)
    print(f"\nAll results saved to {out_csv}")

    # Summary figure (all genes, KM only)
    if len(available) > 1:
        try:
            make_summary_figure(combined)
        except Exception as exc:
            print(f"[WARN] Summary figure failed: {exc}")

    # Compact comparison table
    print(f"\n{'='*60}")
    print("SUMMARY: HC vs Log-rank p-values")
    print(f"{'='*60}")
    print(f"{'Gene':<12} {'Log-rank p':>12} {'HC p':>10} {'HC wins?':>10}")
    print("-" * 46)
    for gene in available:
        sub = combined[combined["gene"] == gene]
        lr_p = sub.loc["Log-rank", "p_value"] if "Log-rank" in sub.index else float("nan")
        hc_p = sub.loc["Higher Criticism (HC)", "p_value"] \
               if "Higher Criticism (HC)" in sub.index else float("nan")
        wins = "YES *" if hc_p < 0.05 and lr_p >= 0.05 else \
               ("both" if lr_p < 0.05 and hc_p < 0.05 else "")
        print(f"{gene:<12} {lr_p:>12.4f} {hc_p:>10.4f} {wins:>10}")


if __name__ == "__main__":
    main()
