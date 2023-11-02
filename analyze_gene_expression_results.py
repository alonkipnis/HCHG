"""
Script to report on the findings of HCHG and log-rank tests applied
to SCANB dataset. The input files are the simulated null results and
results of testing. Both files are obtained via the script 'test_gene_expression.py'.

Example usage:

$python3 illustrate_gene_expression_results.py -null results/SCNAB_null_False_T100_M10000.csv
-results ./results/SCNAB_False_T100.csv -o 'table.csv'

"""

import argparse
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO)
from survival import multi_pvals
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] =  [8, 6]
mpl.style.use('ggplot')
from multitest import MultiTest
from illustrate_gene_expression_survival_curves import illustrate_survival_curve_gene

OUTPUT_DIR_FIGS = "/Users/kipnisal/Dropbox/Apps/Overleaf/Survival Analysis with Sensitivity to Possible Rare and Weak Differences/Figs/"
OUTPUT_DIR_CSV = "/Users/kipnisal/Dropbox/Apps/Overleaf/Survival Analysis with Sensitivity to Possible Rare and Weak Differences/csv/"

#OUTPUT_DIR_FIGS = 'Figs/'
#OUTPUT_DIR_CSV = 'csv/'

SELECTED_GENES = ['FAM20B', 'PBX1', 'IL10RB', 'MBD3', 'MRPS2', 'ANKLE2', 'DDX5', 'MRAS', 'CLCF1', 'MALL']

STATS_TO_DISPLAY = ['hc_greater', 'log_rank_greater', 'logrank_lifelines_fleming-harrington01',
                    'logrank_lifelines_tarone-ware', 'logrank_lifelines_peto']


STATS = ['hc_greater', 'hc_greater_rev', 'log_rank_greater', 'log_rank_greater_rev',
         'logrank_lifelines_None', 'logrank_lifelines_None_rev',
        'logrank_lifelines_wilcoxon', 'logrank_lifelines_wilcoxon_rev',
        'logrank_lifelines_tarone-ware', 'logrank_lifelines_tarone-ware_rev',
        'logrank_lifelines_peto', 'logrank_lifelines_peto_rev',
        'logrank_lifelines_fleming-harrington01', 'logrank_lifelines_fleming-harrington01_rev',
        #'logrank_lifelines_fleming-harrington10', 'logrank_lifelines_fleming-harrington10_rev'
        ]

NAME_NEAT = {'hc_greater': "HCHG",
             'log_rank_greater': 'Log-rank',
        'logrank_lifelines_fleming-harrington01': 'Fleming-Harrington01',
        'logrank_lifelines_fleming-harrington01_rev': 'Fleming-Harrington01',
        'logrank_lifelines_fleming-harrington11': 'Fleming-Harrington11',
        'logrank_lifelines_fleming-harrington11_rev': 'Fleming-Harrington11',
       #'logrank_lifelines_fleming-harrington10': 'Fleming-Harrington10',
       #'logrank_lifelines_fleming-harrington10_rev': 'Fleming-Harrington10',
       'logrank_lifelines_tarone-ware': 'Tarone-Ware',
       'logrank_lifelines_tarone-ware_rev': 'Tarone-Ware',
        'logrank_lifelines_peto': 'Peto-Peto',
        'logrank_lifelines_peto_rev': 'Peto-Peto',
        'logrank_lifelines_wilcoxon': 'Gehan-Wilcoxon',
        'logrank_lifelines_wilcoxon_rev': 'Gehan-Wilcoxon',
        'others': 'Others'
        }

COLOR_LIST = ["tab:red", "tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:green"]
COLOR_LIST_STATS = {'HCHG': 'tab:red',
                    'Log-rank': 'tab:blue',
                    'Fleming-Harrington01': 'tab:orange',
                    #'Fleming-Harrington10': 'tab:orange',
                    'Tarone-Ware': 'tab:purple',
                    'Peto-Peto': 'tab:brown',
                    'Gehan-Wilcoxon': 'tab:green',
                    'Others': 'tab:purple',
                    }
                    

def load_data(data_file_path):
    df = pd.read_csv(data_file_path)
    gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]
    div_probs = df.agg(['mean'])
    thresh = 0.001
    invalid_genes = [g for g in gene_names if np.abs(div_probs[g]['mean'] - 0.5) > thresh]
    df.drop(columns=invalid_genes + ['Unnamed: 0'])
    assert (len(invalid_genes) == 0)
    return df


def qnt(x, q):
    """
    The q-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, q)

def q95(x):
    """
    Compute the 95-th percentile of the vector x
    """
    return qnt(x, 0.95)

def q99(x):
    """
    Compute the 99-th percentile of the vector x
    """
    return qnt(x, 0.99)


def empirical_pval(x, stat_name, df0):
    return np.minimum((np.sum(df0[stat_name].values >= x) ) / len(df0), 1)


def find_pvalues_of_stats_results(df1, df0, stat_name):
    """
    Find the p-values of the statistics in df1 with respect to the null distribution in df0

    """
    val0 =df0[stat_name]
    def stat0(x):
        return np.mean(val0 > x)

    return df1[stat_name].apply(stat0)


def get_discoverable_by_statistic(res:pd.DataFrame, stat, sig_level, side='either'):
    
    if f'{stat}_pvalue' not in res.columns:
        logging.error(f"{stat} not found in results.")
        exit(1)

    if side == 'either':
        a = (res[f'{stat}_pvalue'] <= sig_level)
        if f'{stat}_rev_pvalue' in res.columns:
            a = a | (res[f'{stat}_rev_pvalue'] <= sig_level)
        return a
    
    if side == 'strict':
        if f'{stat}_rev_pvalue' in res.columns:    
            a = (res[f'{stat}_pvalue'] <= sig_level) & (res[f'{stat}_rev_pvalue'] > sig_level)
            b = (res[f'{stat}_pvalue'] > sig_level) & (res[f'{stat}_rev_pvalue'] <= sig_level)
            return a | b
        else:
            return res[f'{stat}_pvalue'] <= sig_level
    if side == 'greater':
        return (res[f'{stat}_pvalue'] <= sig_level)
    if side == 'less':
        return (res[f'{stat}_rev_pvalue'] <= sig_level)

def get_discoverable_by_many_statistics(res:pd.DataFrame, stat_list, sig_level, side='either'):
        "Return an indicator vector of genes discoverable by many statistics"
        discoverable_list = np.array([False] * len(res))
        for stat in stat_list:
            discoverable_list  = discoverable_list | get_discoverable_by_statistic(res, stat, sig_level=sig_level, side=side) 
        return discoverable_list
     

from matplotlib_venn import venn2, venn3

def plot_venn2(set_dict):
    sets = [set_dict[k] for k in set_dict]
    labels = [NAME_NEAT[k] for k in set_dict]
    colors = [COLOR_LIST_STATS[l] for l in labels]
    v = venn2(sets, set_labels = labels, set_colors=colors)

    for i,ch in enumerate('AB'):
        lb = v.get_label_by_id(ch)
        lb.set_color(COLOR_LIST_STATS[labels[i]])
        lb.set_fontsize(20)
    
    for ch in ['10', '01', '11']:
        v.get_label_by_id(ch).set_fontsize(16)


def plot_venn3(set_dict):
    sets = [set_dict[k] for k in set_dict]
    labels = [NAME_NEAT[k] for k in set_dict]
    colors = [COLOR_LIST_STATS[l] for l in labels]
    v = venn3(sets, set_labels = labels, set_colors=colors)

    for i,ch in enumerate('ABC'):
        lb = v.get_label_by_id(ch)
        lb.set_color(COLOR_LIST_STATS[labels[i]])
        lb.set_fontsize(20)
    
    for ch in ['100', '010', '001', '011', '111']:
        v.get_label_by_id(ch).set_fontsize(16)


def report_results_venn_diagram(res:pd.DataFrame, sig_level=0.05):
    # all stats except HC
    
    names = res['name']
    s1 = set(names[get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='strict')])
    s2 = set(names[get_discoverable_by_statistic(res, 'log_rank_greater', sig_level, side='strict')])
    plot_venn2({'hc_greater': s1, 'log_rank_greater': s2})
    filepath = OUTPUT_DIR_FIGS + "venn_HCHG_LR_strict.png"
    plt.savefig(filepath,transparent=True, bbox_inches='tight', pad_inches=0)
    logging.info("Saved figure in " + filepath)
    plt.close()

    # either side one-sided

    s1 = set(names[get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='either')])
    s2 = set(names[get_discoverable_by_statistic(res, 'log_rank_greater', sig_level, side='either')])
    plot_venn2({'hc_greater': s1, 'log_rank_greater': s2})
    filepath = OUTPUT_DIR_FIGS + "venn_HCHG_LR_either.png"
    plt.savefig(filepath,dpi=180, transparent=True, bbox_inches='tight', pad_inches=0)
    logging.info("Saved figure in " + filepath)
    plt.close()


    # all others
    STATS_no_HC = [s for s in STATS if 'hc_' not in s]
    s1 = set(names[get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='either')])
    s2 = set(names[get_discoverable_by_many_statistics(res, STATS_no_HC, sig_level, side='either')])

    plot_venn2({'hc_greater': s1, 'others': s2})

    filepath = OUTPUT_DIR_FIGS + "venn_HCHG_others_either.png"
    plt.savefig(filepath,dpi=180, transparent=True, bbox_inches='tight', pad_inches=0)
    logging.info("Saved figure in " + filepath)
    plt.close()


    # logrank and fleming-harrington01
    s1 = set(names[get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='either')])
    s2 = set(names[get_discoverable_by_statistic(res, 'log_rank_greater', sig_level, side='either')])
    s3 = set(names[get_discoverable_by_statistic(res, 'logrank_lifelines_fleming-harrington01', sig_level, side='either')])

    plot_venn3({'hc_greater': s1,
                'log_rank_greater': s2,
                'logrank_lifelines_fleming-harrington01': s3
                })

    filepath = OUTPUT_DIR_FIGS + "venn_HCHG_others_either_side1.png"
    plt.savefig(filepath,dpi=180, transparent=True, bbox_inches='tight', pad_inches=0)
    logging.info("Saved figure in " + filepath)
    plt.close()

    # logrank and tarone-ware
    s1 = set(names[get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='either')])
    s2 = set(names[get_discoverable_by_statistic(res, 'logrank_lifelines_wilcoxon', sig_level, side='either')])
    s3 = set(names[get_discoverable_by_statistic(res, 'logrank_lifelines_tarone-ware', sig_level, side='either')])

    plot_venn3({'hc_greater': s1,
                'logrank_lifelines_wilcoxon': s2,
                'logrank_lifelines_tarone-ware': s3
                })

    filepath = OUTPUT_DIR_FIGS + "venn_HCHG_others_either_side2.png"
    plt.savefig(filepath, dpi=180, transparent=True, bbox_inches='tight', pad_inches=0)
    logging.info("Saved figure in " + filepath)
    plt.close()



def report_results_HC_logrank(res:pd.DataFrame, sig_level=0.05):

    log_rank_non = (res[f'log_rank_greater_pvalue'] > sig_level) & (res[f'log_rank_greater_rev_pvalue'] > sig_level)
    log_rank_1side_strict = get_discoverable_by_statistic(res, 'log_rank_greater', sig_level, side='strict')
    log_rank_1side_either = get_discoverable_by_statistic(res, 'log_rank_greater', sig_level, side='either')

    hc_non = (res['hc_greater_pvalue'] > sig_level) & (res['hc_greater_rev_pvalue'] > sig_level)
    hc_1side_strict = get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='strict')
    hc_1side_either = get_discoverable_by_statistic(res, 'hc_greater', sig_level, side='either')


    print("Strictly one-sided effect:")
    print("\tDiscoverable by HCHG: ", np.sum(hc_1side_strict))
    print(f"\tDiscoverable by Logrank: ", np.sum(log_rank_1side_strict))

    print(f"\tDiscoverable by HCHG and LR: ", np.sum(hc_1side_strict & log_rank_1side_strict))
    print(f"\tDiscoverable by HCHG but not LR: ", np.sum(hc_1side_strict & (~log_rank_1side_strict) )   )
    print(f"\tDiscoverable by LR but not HCHG: ", np.sum((~hc_1side_strict) & ( log_rank_1side_strict) )   )
    print(f"\tDiscoverable by neither HCHG nor LR: ", np.sum((~hc_1side_strict) & (~log_rank_1side_strict) )   )

    print(" Either side effect:")

    print("\tDiscoverable by HCHG: ", np.sum(hc_1side_either))
    print(f"\tDiscoverable by LR: ", np.sum(log_rank_1side_either))

    print(f"\tDiscoverable by HCHG and LR: ", np.sum(hc_1side_either & log_rank_1side_either))
    print(f"\tDiscoverable by HCHG but not LR: ", np.sum(hc_1side_either & (~log_rank_1side_either) )   )
    print(f"\tDiscoverable by LR but not HCHG: ", np.sum((~hc_1side_either) & ( log_rank_1side_either) )   )
    print(f"\tDiscoverable by neither HCHG nor LR: ", np.sum((~hc_1side_either) & (~log_rank_1side_either) )   )

    # all stats except HC
    STATS_no_HC = [s for s in STATS if 'hc_' not in s]
    all_two_sided = get_discoverable_by_many_statistics(res, STATS_no_HC, sig_level, side='either')

    print("All stats except HC (Either side effect):")
    print(f"\tDiscoverable by all others: ", np.sum(all_two_sided))
    print("HCHG but not others (Either side effect):")
    print(f"\tDiscoverable by HCHG but not by any other: ", np.sum(hc_1side_either & (~all_two_sided)))


def report_discoveries_all_stats_multiple_siglevels(res:pd.DataFrame):
    """
    Report number of discoveries by each statistic at multiple significance levels

    Args:
        res:  dataframe containing the P-values of every statistic in the list STATS
    """
    results = []
    sig_levels = [0.05, 0.01, 0.001, 0.0001]
    for sig_level in sig_levels:
        for stn in STATS:
            if "_rev" not in stn:
                sig = sig_level
                nod = np.sum( get_discoverable_by_statistic(res, stn, sig_level=sig, side='either') )
                results.append({'sig_level': sig_level, 'number of discoveries': nod, 'stat_name': stn})

    return pd.DataFrame(results)


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


def find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True, gamma=.4):
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    mt = MultiTest(pvals[pvals <= 1], stbl=stbl)
    _, hct = mt.hc(gamma=gamma)
    return pvals <= hct


def prepare_for_display(resi_disp):
    df = resi_disp.copy()
    flip_idc = df['hc_greater'] < df['hc_greater_rev']
    df['flip'] = '$>$ med'
    df.loc[flip_idc, 'flip'] = '$<$ med'
    lo_stats_to_display = []
    for stat_name in STATS_TO_DISPLAY:
        df.loc[:, NAME_NEAT[stat_name]] = np.minimum(df[stat_name+'_pvalue'], df[stat_name+'_rev_pvalue'])
        lo_stats_to_display.append(NAME_NEAT[stat_name])
        
    rr = df.reset_index().filter(['name', 'flip'] + lo_stats_to_display)

    return rr

def get_pvals_for_gene(resi_disp, gene_name):
    rr = []
    for stat in STATS_TO_DISPLAY:
        r = resi_disp[resi_disp.name == gene_name]
        if len(r) == 0:
            print(f"Error: could not find gene {gene_name}")
        rr.append({'stat': stat, 'p-value': r[stat + '_pvalue'].values[0]})
    return pd.DataFrame(rr).set_index('stat')

def main():
    parser = argparse.ArgumentParser(description='Illustrate Results')
    parser.add_argument('-null', type=str, help='null data file (csv)')
    parser.add_argument('-results', type=str, help='results file (csv)')
    parser.add_argument('-o', type=str, help='output table', default=OUTPUT_DIR_CSV + "/results_SCANB.csv")
    
    #parser.add_argument('--illustrate', action='store_true', help='illustrate survival curves')
    #parser.add_argument('-data', type=str, help='raw gene expression data', default=OUTPUT_DIR_CSV + "Data/SCANB_groups_valid.csv")
    args = parser.parse_args()
    #

    assert args.null is not None, "Please proivde null data file."
    assert args.results is not None, "Please proivde results data file."

    logging.info(f"Reading from {args.results}...")
    res = pd.read_csv(args.results)
    gene_names = list(res.name.unique())
    logging.info(f"Found {len(gene_names)} unique genes.")

    logging.info(f"Reading null simulation results from {args.null}...")
    df0 = pd.read_csv(args.null).filter(regex='^((?!Unnamed).)*$')
    
    sig_level = 0.05
    crit_vals = df0.agg([lambda x : qnt(x, 1 - sig_level) ]).filter(
        ['log_rank_greater', 
         'hc_greater', 
         'hc_greater_rev', 
         'log_rank_greater_rev'])
    logging.info(f"Critical test values at significance level {sig_level} from simulated null data:")
    logging.info(crit_vals)
    
    for stat_name in STATS:
        res.loc[:, stat_name + '_pvalue'] = find_pvalues_of_stats_results(res, df0, stat_name)

    print(report_discoveries_all_stats_multiple_siglevels(res))
    
    report_results_HC_logrank(res, sig_level=sig_level)

    report_results_venn_diagram(res, sig_level=sig_level)

    resi = arrange_results_for_presentation(df0, res)
    resi_disp = resi[resi.name.isin(SELECTED_GENES)]
    df_disp = prepare_for_display(resi_disp).set_index('name')
    print(df_disp)
    df_disp.to_csv(args.o)
    logging.info(f"Saved table in {args.o}")

    for gene_name in SELECTED_GENES:
        df_pvals = get_pvals_for_gene(resi_disp, gene_name)
        df_pvals.to_csv(f"{OUTPUT_DIR_CSV}{gene_name}_pvals.csv")
        logging.info(f"Saved p-values for {gene_name} in {OUTPUT_DIR_CSV}{gene_name}_pvals.csv")

    
if __name__ == '__main__':
    main()

