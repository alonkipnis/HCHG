"""
Script to illustrate survival curves of certain genes from SCANB dataset.
For each gene in the list, the script produces Kaplan-Meir survival curves of both groups,
indicates time instances suspected of having an excessive risk, and list those times in a table.

The inputs are:
 1. SCANB dataset with each gene indicating its group membership (1 or 0)
 2. Output folder

Example usage:

$python3 illustrate_gene_expression_survival_curves.py -data Data/SCANB_groups_valid.csv
-outdir Figs/

"""

SELECTED_GENES = ['SIGMAR1', 'ST6GALNAC5', 'DCK', 'ADSS', 'KCTD9', 'VAMP4', 'HIST1H3G', 'TMEM38B', 'SIGMAR1', 'SMG9',
                'FBXL12', 'PDE6D', 'BTNL8']

import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

from test_gene_expression import reduce_time_resolution, two_groups_gene
from survival import evaluate_test_stats, multi_pvals
from multitest import MultiTest
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [12, 6]
mpl.style.use('ggplot')


def empirical_pval(x, stat_name, df0):
    return np.minimum((np.sum(df0[stat_name].values >= x) ) / len(df0), 1)


def load_data(data_file_path, ):
    df = pd.read_csv(data_file_path)
    gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]
    div_probs = df.agg(['mean'])
    thresh = 0.001
    invalid_genes = [g for g in gene_names if np.abs(div_probs[g]['mean'] - 0.5) > thresh]
    df.drop(columns=invalid_genes + ['Unnamed: 0'])
    assert (len(invalid_genes) == 0)
    return df


def find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True, gamma=.5):
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    mt = MultiTest(pvals[pvals <= 1], stbl=stbl)
    hc, hct = mt.hc(gamma=gamma)
    return pvals <= hct


def illustrate_survival_curve_gene(df, gene_name, T, stbl=False,
                               show_HCT=True, randomize_HC=False):
    dfg = reduce_time_resolution(two_groups_gene(df, gene_name), T)

    Nt1, Nt2 = dfg['at-risk1'].values, dfg['at-risk2'].values
    Ot1, Ot2 = dfg['dead1'].values, dfg['dead2'].values

    stats = evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, stbl=stbl, randomize=randomize_HC)
    stats_rev = evaluate_test_stats(Nt2, Nt1, Ot2, Ot1, stbl=stbl, randomize=randomize_HC)
    if stats['hc_greater'] < stats_rev['hc_greater']:  # reverse groups
        dfg = dfg.rename(columns={'at-risk1': 'at-risk2', 'at-risk2': 'at-risk1',
                                  'dead1': 'dead2', 'dead2': 'dead1',
                                  'censored1': 'censored2', 'censored2': 'censored1'
                                  })
        temp = stats
        stats = stats_rev
        stats_rev = temp

    Nt1, Nt2 = dfg['at-risk1'].values, dfg['at-risk2'].values
    Ot1, Ot2 = dfg['dead1'].values, dfg['dead2'].values

    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    pvals_rev = multi_pvals(Nt2, Nt1, Ot2, Ot1, randomize=False)
    fpval = find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True)

    dfg['pvalue'] = pvals
    dfg['pvalue_rev'] = pvals_rev
    cumc1 = dfg['censored1'].cumsum()
    cumc2 = dfg['censored2'].cumsum()
    dfg['Survival Proportion X'] = (dfg['at-risk1'] - dfg['censored1']) / (dfg['at-risk1'].max() - cumc1)
    dfg['Survival Proportion Y'] = (dfg['at-risk2'] - dfg['censored2']) / (dfg['at-risk2'].max() - cumc2)
    # dfg['censored1'] = dfg['at-risk1'] - dfg['']

    df_disp = dfg[fpval].rename(columns={'at-risk1': 'at-risk X', 'at-risk2': 'at-risk Y',
                                         'dead1': 'events X', 'dead2': 'events Y'
                                         })

    plt.step(dfg.index, dfg['Survival Proportion X'], 'b', where='pre')
    plt.step(dfg.index, dfg['Survival Proportion Y'], 'r', where='pre')
    ct1 = dfg['censored1'] > 0
    ct2 = dfg['censored2'] > 0
    s1 = 10 * (dfg.loc[ct1, 'censored1'].max() / dfg.loc[ct1, 'censored1']).values
    s2 = 10 * (dfg.loc[ct2, 'censored2'].max() / dfg.loc[ct2, 'censored2']).values
    plt.scatter(dfg.index[ct1], dfg.loc[ct1, 'Survival Proportion X'],
                marker='+', c='b',
                s=s1, alpha=.5)
    plt.scatter(dfg.index[ct2], dfg.loc[ct2, 'Survival Proportion Y'],
                marker='+', c='r',
                s=s2, alpha=.5)

    plt.legend([r'$\hat{S}_x$', r'$\hat{S}_y$'], fontsize=16, loc=1)

    if show_HCT:
        plt.bar(dfg.index[:len(fpval)], fpval, color='k', alpha=.2, width=.5)

    plt.title(f"{gene_name}, (HC={np.round(stats['hc_greater'],2)}, HCrev={np.round(stats_rev['hc_greater'],2)} Log-rank={np.round(stats['log_rank_greater'],2)})")
    plt.ylabel('proportion', fontsize=16)
    plt.xlabel(r'$t$ [Time]', fontsize=16)
    plt.ylim([0.7, 1.01])

    return df_disp, dfg


def main():
    parser = argparse.ArgumentParser(description='Illustrate Results')
    parser.add_argument('-data', type=str, help='SCANB gene expression data file',
                        default="data/SCANB_groups_valid.csv")
    parser.add_argument('-T', type=int, help='number of intervals')
    parser.add_argument('-gene-names', type=list, help='list of gene names')
    parser.add_argument('-outdir', type=str, help='output directory for images',
                        default="./")

    args = parser.parse_args()
    #

    outdir = args.outdir
    logging.info(f"Reading data from {args.data}...")
    df = load_data(args.data)

    lo_genes = args.gene_names
    T = args.T

    for gene_name in lo_genes:

        fig_filename = outdir + gene_name + ".png"
        plt.figure()
        df_disp, dfp = illustrate_survival_curve_gene(df, gene_name, T, stbl=False)
        logging.info(f"Writing survival curve to {fig_filename}.")
        plt.savefig(fig_filename)

        dfd = df_disp.copy()
        dfd = dfd.iloc[:, :-2]

        for tag in ['pvalue', 'pvalue_rev']:
            dfd[tag] = np.round(dfd[tag], 3)

        for tag in ['at-risk X', 'at-risk Y', 'events X', 'events Y']:
            dfd[tag] = dfd[tag].astype(int)

        dfd.index.name = 'time'

        table_filename = outdir + f'{gene_name}.csv'
        logging.info(f"Writing table of suspected time instances to {table_filename}.")
        dfd.filter(['at-risk X', 'at-risk Y', 'events X', 'events Y', 'pvalue', 'pvalue_rev']) \
            .to_csv(table_filename)


if __name__ == '__main__':
    main()




