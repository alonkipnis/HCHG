
"""
Script to report on the findings of HCHG and log-rank tests applied
to SCANB dataset. The input files are the simulated null results and
results of testing. Both files are obtained via the script 'test_gene_expression.py'.

Example usage:

$python3 illustrate_gene_expression_results.py -null results/SCNAB_null_False_T100_M10000.csv
-results ./results/SCNAB_False_T100.csv -o 'table.csv'

"""

SELECTED_GENES = ['SIGMAR1', 'ST6GALANC5', 'DCK', 'ADSS', 'KCTD9',
                      'VAMP4', 'HIST1H3G', 'TMEM38B', 'SIGMAR1', 'SMG9',
                       'FBXL12', 'PDE6D', 'BTNL8', 'TRPS1']
SELECTED_GENES = ['DCK', 'EIF2B4', 'ADSS', 'OSGEP', 'TMCO1', 'USP21', 'NINL', 'SCAP',
        'INTS3', 'SSR3', 'AGT']

import argparse
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO)
from survival import (q95, multi_pvals, evaluate_test_stats)
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] =  [8, 6]
mpl.style.use('ggplot')
from multitest import MultiTest
from tqdm import tqdm
from illustrate_gene_expression_survival_curves import illustrate_survival_curve_gene


def empirical_pval(x, stat_name, df0):
    return np.minimum((np.sum(df0[stat_name].values >= x) ) / len(df0), 1)


def find_pvalues_of_stats_results(df1, df0, stat_name):
    val0 =df0[stat_name]
    def stat0(x):
        return np.mean(val0 > x)

    return df1[stat_name].apply(stat0)



def report_results(res:pd.DataFrame, sig_level=0.05):
    log_rank_non = (res['log_rank_greater_pvalue'] > sig_level) & (res['log_rank_greater_rev_pvalue'] > sig_level)
    log_rank_1side_strict = (res['log_rank_greater_pvalue'] <= sig_level) & (res['log_rank_greater_rev_pvalue'] > sig_level)
    log_rank_1side_strict_rev = (res['log_rank_greater_pvalue'] > sig_level) & (res['log_rank_greater_rev_pvalue'] <= sig_level)
    log_rank_2side = (res['log_rank_greater_pvalue'] <= sig_level) | (res['log_rank_greater_rev_pvalue'] <= sig_level)

    log_rank_strict = log_rank_1side_strict | log_rank_1side_strict_rev

    hc_non = (res['hc_greater_pvalue'] > sig_level) & (res['hc_greater_rev_pvalue'] > sig_level)
    hc_1side_strict = (res['hc_greater_pvalue'] <= sig_level) & (res['hc_greater_rev_pvalue'] > sig_level)
    hc_1side_strict_rev = (res['hc_greater_pvalue'] > sig_level) & (res['hc_greater_rev_pvalue'] <= sig_level)
    hc_2side = (res['hc_greater_pvalue'] <= sig_level) | (res['hc_greater_rev_pvalue'] <= sig_level)

    hc_strict = hc_1side_strict | hc_1side_strict_rev

    print("Strictly one-sided effect:")
    print("\tDiscoverable by HC: ", np.sum(hc_strict))
    print("\tDiscoverable by LR: ", np.sum(log_rank_strict))
    print("\tDiscoverable by HC and LR: ", np.sum(hc_strict & log_rank_strict))
    print("\tDiscoverable by HC but not LR: ", np.sum(hc_strict & (1 - log_rank_strict) )   )
    print("\tDiscoverable by LR but not HC: ", np.sum((1 - hc_strict) & ( log_rank_strict) )   )
    print("\tDiscoverable by neigher HC nor LR: ", np.sum((1 - hc_strict) & (1 - log_rank_strict) )   )


    print(" Either side effect:")
    print("\tDiscoverable by HC: ", np.sum(hc_2side))
    print("\tDiscoverable by LR: ", np.sum(log_rank_2side))
    print("\tDiscoverable by HC and LR: ", np.sum(hc_2side & log_rank_2side))
    print("\tDiscoverable by HC but not LR: ", np.sum(hc_2side & (1 - log_rank_2side) )   )
    print("\tDiscoverable by LR but not HC: ", np.sum((1 - hc_2side) & ( log_rank_2side) )   )
    print("\tDiscoverable by neigher HC nor LR: ", np.sum((1 - hc_2side) & (1 - log_rank_2side) )   )



def report_results_critvals(res:pd.DataFrame, HC_critval, logrank_critval):
    """
    Print number of discoveries by each test and report

    Args:
        res:  results file
        crit_vals:  critical test values

    Returns:
        None

    """

    LRt = logrank_critval
    HCt = HC_critval

    log_rank_1side_strict = (res.log_rank_greater > LRt) & (res.log_rank_greater_rev < LRt)
    log_rank_1side_strict_rev = (res.log_rank_greater < LRt) & (res.log_rank_greater_rev > LRt)
    log_rank_2side = (res.log_rank_greater > LRt) | (res.log_rank_greater_rev > LRt)

    log_rank_strict = log_rank_1side_strict | log_rank_1side_strict_rev

    hc_1side_strict = (res.hc_greater > HCt) & (res.hc_greater_rev < HCt)
    hc_1side_strict_rev = (res.hc_greater < HCt) & (res.hc_greater_rev > HCt)
    hc_2side = (res.hc_greater > HCt) | (res.hc_greater_rev > HCt)

    hc_strict = hc_1side_strict | hc_1side_strict_rev

    print("Strictly one-sided effect:")
    print("\tDiscoverable by HC: ", np.sum(hc_strict))
    print("\tDiscoverable by LR: ", np.sum(log_rank_strict))

    print("\tDiscoverable by HC and LR: ", np.sum(hc_strict & log_rank_strict))
    print("\tDiscoverable by HC but not LR: ", np.sum(hc_strict & (1 - log_rank_strict) )   )
    print("\tDiscoverable by LR but not HC: ", np.sum((1 - hc_strict) & ( log_rank_strict) )   )
    print("\tDiscoverable by neigher HC nor LR: ", np.sum((1 - hc_strict) & (1 - log_rank_strict) )   )

    print(" Either side effect:")

    print("\tDiscoverable by HC: ", np.sum(hc_2side))
    print("\tDiscoverable by LR: ", np.sum(log_rank_2side))


    print("\tDiscoverable by HC and LR: ", np.sum(hc_2side & log_rank_2side))
    print("\tDiscoverable by HC but not LR: ", np.sum(hc_2side & (1 - log_rank_2side) )   )
    print("\tDiscoverable by LR but not HC: ", np.sum((1 - hc_2side) & ( log_rank_2side) )   )
    print("\tDiscoverable by neigher HC nor LR: ", np.sum((1 - hc_2side) & (1 - log_rank_2side) )   )


def arrange_results_for_presentation(df0, res):
    # use minimal P-value between both sides
    res['log_rank_pval'] = [empirical_pval(v, 'log_rank_greater', df0) for v in res['log_rank_greater'].values]
    res['log_rank_rev_pval'] = [empirical_pval(v, 'log_rank_greater_rev', df0) for v in res['log_rank_greater_rev'].values]
    res['hc_greater_pval'] = [empirical_pval(v, 'hc_greater', df0) for v in res['hc_greater'].values]
    res['hc_greater_rev_pval'] = [empirical_pval(v, 'hc_greater_rev', df0) for v in res['hc_greater_rev'].values]
    res['hc_pval'] = np.minimum(res['hc_greater_pval'], res['hc_greater_rev_pval'])
    res['log_rank_pval'] = np.minimum(res['log_rank_pval'], res['log_rank_rev_pval'])
    res['l-ratio'] = res['hc_pval'] / res['log_rank_pval']
    return res


def find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True, gamma=.5):
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    mt = MultiTest(pvals[pvals <= 1], stbl=stbl)
    hc, hct = mt.hc(gamma=gamma)
    return pvals <= hct


def find_pvalues_of_stats_results(df1, df0, stat_name):
    val0 =df0[stat_name]
    def stat0(x):
        return np.mean(val0 > x)

    return df1[stat_name].apply(stat0)


def prepare_for_display(resi_disp):
    flip_idc = resi_disp['hc_greater'] < resi_disp['hc_greater_rev']
    resi_disp['flip'] = '$>$ med'
    resi_disp.loc[flip_idc, 'flip'] = '$<$ med'
    resi_disp['hc'] = np.maximum(resi_disp['hc_greater'], resi_disp['hc_greater_rev'])
    resi_disp['log_rank'] = np.maximum(resi_disp['log_rank_greater'], resi_disp['log_rank_greater_rev'])
    rr = resi_disp.reset_index().filter(['name', 'hc', 'hc_pval', 'log_rank', 'log_rank_pval', 'flip'])

    return rr


def qnt(x, q):
    """
    The q-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, q)


def main():
    parser = argparse.ArgumentParser(description='Illustrate Results')
    parser.add_argument('-null', type=str, help='null data', default="results/SCANB_null_False_T100_M1.csv")
    parser.add_argument('-results', type=str, help='results', default="results/SCANB_greater_False_T100.csv")
    parser.add_argument('-o', type=str, help='output table', default="table.csv")
    args = parser.parse_args()
    #

    logging.info(f"Reading from {args.results}...")
    res = pd.read_csv(args.results)
    gene_names = list(res.name.unique())
    logging.info(f"Found {len(gene_names)} unique genes.")

    logging.info(f"Reading null simulation results from {args.null}...")
    df0 = pd.read_csv(args.null).filter(regex='^((?!Unnamed).)*$')
    
    sig_level = 0.01
    crit_vals = df0.agg([lambda x : qnt(x, 1 - sig_level) ]).filter(
        ['log_rank_greater', 
         'hc_greater', 
         'hc_greater_rev', 
         'log_rank_greater_rev'])
    logging.info(f"Computed the following critical test values at significance level {sig_level}:")
    logging.info(crit_vals)
    
    
    for stat_name in ['hc_greater', 'log_rank_greater', 'hc_greater_rev', 'log_rank_greater_rev']:
        res.loc[:, stat_name + '_pvalue'] = find_pvalues_of_stats_results(res, df0, stat_name)
        
    report_results(res, sig_level=sig_level)

    resi = arrange_results_for_presentation(df0, res)
    resi_disp = resi[resi.name.isin(SELECTED_GENES)]
    df_disp = prepare_for_display(resi_disp).set_index('name')
    print(df_disp)
    df_disp.to_csv(args.o)
    logging.info(f"Saved table in {args.o}")


if __name__ == '__main__':
    main()

