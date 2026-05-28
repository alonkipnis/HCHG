"""
Download and prepare immuno-oncology clinical trial data from the kmdata R package.

The kmdata package (github.com/raredd/kmdata) contains reconstructed individual
patient-level data (IPD) from 153 phase III oncology trials, reverse-engineered
from published Kaplan-Meier curves using the Guyot algorithm.

Trials downloaded by default:
  POPLAR     - Atezolizumab vs docetaxel in 2nd-line NSCLC (Fehrenbacher 2016)
  CheckMate_066  - Nivolumab vs dacarbazine in metastatic melanoma (Robert 2015)

Output:  AppNote/data/<TRIAL>.csv
Columns: trial-specific, but typically includes time, status/event, arm columns.

Usage:
    python 01_get_kmdata.py [--trials POPLAR CheckMate_066]

Requires:
    R (>= 4.0) accessible as 'Rscript'
    Internet access (for kmdata package installation from GitHub)
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Trials with well-documented delayed treatment effect / crossing survival curves
DEFAULT_TRIALS = [
    "POPLAR",         # Atezolizumab vs docetaxel, NSCLC
    "CheckMate_066",  # Nivolumab vs dacarbazine, melanoma
    "CheckMate_057",  # Nivolumab vs docetaxel, non-squamous NSCLC
]

R_EXPORT_SCRIPT = r"""
suppressPackageStartupMessages({
  if (!require("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  if (!require("kmdata", quietly = TRUE)) {
    cat("Installing kmdata from GitHub (raredd/kmdata)...\n")
    remotes::install_github("raredd/kmdata", quiet = TRUE)
  }
  library(kmdata)
})

trials  <- commandArgs(trailingOnly = TRUE)
out_dir <- trials[length(trials)]   # last arg is output dir
trials  <- trials[-length(trials)]  # remaining args are trial names

cat(sprintf("kmdata contains %d trials.\n", length(kmdata)))
cat(sprintf("Requested: %s\n", paste(trials, collapse = ", ")))

for (trial in trials) {
  if (!trial %in% names(kmdata)) {
    # Fuzzy match: try replacing _ with . or space
    alt <- names(kmdata)[grepl(gsub("_", ".", trial, fixed = TRUE),
                               names(kmdata), ignore.case = TRUE)]
    if (length(alt) == 0) {
      cat(sprintf("[SKIP] '%s' not found in kmdata. Available:\n", trial))
      cat(paste(sort(names(kmdata)), collapse = "\n"), "\n")
      next
    }
    trial <- alt[1]
    cat(sprintf("[INFO] Using '%s' as match.\n", trial))
  }
  df   <- kmdata[[trial]]
  fn   <- file.path(out_dir, paste0(trial, ".csv"))
  write.csv(df, fn, row.names = FALSE)
  cat(sprintf("[OK]   %s -> %s  (%d rows, cols: %s)\n",
              trial, fn, nrow(df), paste(colnames(df), collapse = ", ")))
}
"""


def run_r_export(trials: list[str], out_dir: Path) -> bool:
    """Call Rscript to install kmdata and export trials to CSV.

    Returns True on success.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write R script to a temp file (avoids shell quoting issues)
    r_script_path = out_dir / "_export_kmdata.R"
    r_script_path.write_text(R_EXPORT_SCRIPT)

    cmd = ["Rscript", "--no-save", str(r_script_path)] + trials + [str(out_dir)]
    print("Running R export (may take a few minutes on first run)...")
    result = subprocess.run(cmd, text=True, capture_output=True)

    print(result.stdout)
    if result.returncode != 0:
        print("R stderr:", result.stderr[-2000:], file=sys.stderr)
        return False
    return True


def check_output(trials: list[str], out_dir: Path) -> None:
    """Print a summary of exported files."""
    import pandas as pd

    print("\n--- Downloaded files ---")
    for trial in trials:
        fp = out_dir / f"{trial}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            print(f"  {trial}: {len(df)} patients, columns: {list(df.columns)}")
        else:
            print(f"  {trial}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Download kmdata trial IPD to CSV")
    parser.add_argument("--trials", nargs="+", default=DEFAULT_TRIALS,
                        help="Trial names (as in kmdata R package)")
    parser.add_argument("--out-dir", default=str(DATA_DIR),
                        help="Output directory for CSV files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ok = run_r_export(args.trials, out_dir)
    if ok:
        check_output(args.trials, out_dir)
    else:
        print("\nR export failed. Possible causes:")
        print("  - R not installed or not on PATH")
        print("  - GitHub install blocked (no internet / firewall)")
        print("  - kmdata package renamed or moved")
        print("\nAlternative: install R package manually and re-run:")
        print("  Rscript -e \"remotes::install_github('raredd/kmdata')\"")
        sys.exit(1)


if __name__ == "__main__":
    main()
