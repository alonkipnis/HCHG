# Higher criticism for rare and weak non‑proportional hazard deviations in survival analysis

Code to reproduce the simulations, figures, and analyses from:

Kipnis, Alon, Ben Galili, and Zohar Yakhini. Higher criticism for rare and weak non‑proportional hazard deviations in survival analysis. Biometrika (2025): asaf075.


## Overview

This repository has been cleaned to include only the code needed to reproduce the paper’s results:

- Synthetic data demos and visuals
- Experimental array illustrating the phase transition
- SCANB dataset demonstration

Auxiliary or unrelated materials have been moved to `archive/` and are excluded from the repository.


## Software package

The `lifelines-hc` Python package (the `higher_criticism_test` / HCHG
implementation and the Application Note analysis code) lives in its own
repository: **https://github.com/alonkipnis/lifelines-hc**
(`pip install lifelines-hc`).


## Environment

Python 3.8+ is recommended. On Apple Silicon macOS, prefer a native arm64 environment (e.g., conda/mamba) to avoid binary wheel mismatches.

Install dependencies:

```
pip install -r requirements.txt
```

Conda example (recommended on macOS ARM):
```
mamba create -n survival-hchg python=3.10
conda activate survival-hchg
pip install -r requirements.txt
```


## Repository layout (kept for reproduction)

- `survival.py` — core HCHG routines (evaluation, statistics, helpers)
- `phase_transition_experiment/` — phase transition experiment array
  - `configurations.py`, `params.yaml` — configuration generation
  - `para_run.py` — runs the array locally or via Dask
  - `illustrate_phase_diagram.py` — plots phase diagrams from results
- Synthetic demos
  - `illustrate_survival_curve.py` — survival curve visualization from counts
  - `plot_survival_curve_example_with_censoring.py` — Figure 1 synthetic example
  - `synthetic_experiment.py` — synthetic experiments and summary tables
  - `illustrate_survival_curve.ipynb`, `synthetic_experiment.ipynb` — companion notebooks
- SCANB demo
  - `test_gene_expression.py` — null and alternative evaluations over SCANB
  - `analyze_gene_expression_results.py` — summary tables and plots
  - `illustrate_gene_expression_survival_curves.py` — per‑gene visuals and tables
  - `Data/SCANB_groups*.csv` — SCANB groupings and censored variant
  - `csv/` — auxiliary per‑gene CSVs and outputs
- Outputs
  - `results/` — generated CSVs (ignored by git)
  - `Figs/` — generated figures (example output directory)


## How to reproduce

1) Synthetic visuals and experiments

- Figure 1 example (survival curve with censoring):
  ```
  python plot_survival_curve_example_with_censoring.py
  ```

- Synthetic experiment tables and summaries (writes into `results/`):
  ```
  python synthetic_experiment.py
  ```

2) Phase transition experiment array

- Run the array (locally):
  ```
  python phase_transition_experiment/para_run.py -p phase_transition_experiment/params.yaml -o results_phase.csv
  ```

- Plot phase diagrams from results (into `Figs/`):
  ```
  python phase_transition_experiment/illustrate_phase_diagram.py -i results_phase.csv -o Figs/
  ```

3) SCANB demonstration

- Evaluate null and alternative across SCANB and save outputs (see in‑script args for options):
  ```
  python test_gene_expression.py -i Data/SCANB_groups_valid_KS_censored.csv -o SCANB
  ```

- Analyze and aggregate results:
  ```
  python analyze_gene_expression_results.py -o results/SCANB_analyzed_results.csv
  ```

- Visualize per‑gene survival curves and suspected time instances (writes to `Figs/` and `csv/`):
  ```
  python illustrate_gene_expression_survival_curves.py -data Data/SCANB_groups_valid.csv -T 100 -gene-names GENE1 GENE2 -outdir ""
  ```


## Notes

- Heavy outputs and unrelated materials are under `archive/` (ignored by git).
- `results/` is ignored by git and used by scripts for generated CSVs.
- The SCANB data files listed under `Data/` are required inputs; see `Data/data_link.txt` for provenance.
