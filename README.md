# Detecting rare and weak deviations of non-proportional hazard for survival analysis

This repository contains code for simulation and analysis reported in the article:

Ben Galili, Alon Kipnis, and Zohar Yakhini. "Detecting rare and weak deviations of non-proportional hazard for survival analysis." 2025. https://arxiv.org/pdf/2310.00554


## The file ``survivial.py`` is the main file containing main HCHG functions

## Instructions to reproduce article results:

### Synthetic data experimens:
- Results in Table 1 are generated using ``Synthetic_experiment.py`` 
- Data for Figure 1 is generated using ``plot_survival_curve_example_with_censoring.py``
- Asymptotic power analysis results are generated using ``phase_transition_experiments/para_run.py``. This script calls the function ``evaluate_rare_and_weak`` from ``survival.py`` for each configuration. Configurations are specified in the file `phase_transition_experiments/configurations``. You can use the file ``phase_transition_experiments/illustrate_phase_diagram.py`` to illustrate the phase diagrams (Figure 3 and 4)

#### Real data analysis:
- Use the ``test_gene_expression.py``. You should use this file once to evalaute null the distribution for Hypergeometric P-values and again to evalaute non-null results. 
- Use the file ``analyze_gene_expression_results.py`` to get the results reported in Table 4 and Figure 6. 
- USe the file ``illustrate_gene_expression_survival_curves.py`` to get the tables and surivial curves of Figure 7. 

