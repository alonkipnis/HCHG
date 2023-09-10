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

SELECTED_GENES = ['ERGIC2', 'EPB41L4B', 'TOR1AIP2']


import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

from test_gene_expression import reduce_time_resolution, two_groups_table
from survival import evaluate_test_stats, multi_pvals
from multitest import MultiTest
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import KaplanMeierFitter

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
    _, hct = mt.hc(gamma=gamma)
    return pvals <= hct


def plot_survival_curve_time2event(df):
    # Fit & plot leukemia dataset
    kmf1 = KaplanMeierFitter()
    df1  = df[df['group'] == 0]
    kmf1.fit(df1['time'], df1['event'], label='$\hat{S}_x$')
    ax = kmf1.plot(ci_show = False, show_censors=True)

    kmf2 = KaplanMeierFitter()
    df2  = df[df['group'] >= 1]
    kmf2.fit(df2['time'], df2['event'], label='$\hat{S}_y$')
    ax = kmf2.plot(ci_show = False, show_censors=True)
    plt.legend(fontsize=18)


def illustrate_survival_curve_time2event(df_time2event,
                                        stbl=True, # type of HC stat
                               show_HCT=True, randomize_HC=False,
                               show_stats_in_title=True, flip_sides=False):
    
    
     # to table representation
    dfg = two_groups_table(df_time2event, 'group')

    Nt1, Nt2 = dfg['at_risk:0'].values, dfg['at_risk:1'].values
    Ot1, Ot2 = dfg['observed:0'].values, dfg['observed:1'].values

    stats = evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, stbl=stbl, randomize=randomize_HC)
    stats_rev = evaluate_test_stats(Nt2, Nt1, Ot2, Ot1, stbl=stbl, randomize=randomize_HC)
    if flip_sides and (stats['hc_greater'] < stats_rev['hc_greater']): # reverse groups
        dfg = dfg.rename(columns={'at_risk:0': 'at_risk:1', 'at_risk:1': 'at_risk:0',
                                'observed:0': 'observed:1', 'observed:1': 'observed:0',
                                'censored:0': 'censored:1', 'censored:1': 'censored:0'
                                })
        temp = stats
        stats = stats_rev
        stats_rev = temp
        Nt1, Nt2 = dfg['at_risk:0'].values, dfg['at_risk:1'].values
        Ot1, Ot2 = dfg['observed:0'].values, dfg['observed:1'].values
        logging.info("Flipped sides")

    plot_survival_curve_time2event(df_time2event)
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    pvals_rev = multi_pvals(Nt2, Nt1, Ot2, Ot1, randomize=False)
    
    dfg['pvalue'] = pvals
    dfg['pvalue_rev'] = pvals_rev
    

    fpval = find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True)
    df_disp = dfg[fpval].rename(columns={'at_risk:0': 'at_risk X', 'at_risk:1': 'at_risk Y',
                                         'observed:0': 'observed X', 'observed:1': 'observed Y'
                                         })
    
    if show_HCT:
        plt.bar(dfg.index[:len(fpval)], fpval, color='k', alpha=.2, width=.5)
        
    if show_stats_in_title:
        stats_str = f"HC={np.round(stats['hc_greater'],2)}, Log-rank={np.round(stats['log_rank_greater'],2)}"
        T = df_time2event['time'].max()
        plt.text(int(T/3), 1, stats_str, backgroundcolor='white')
    plt.ylabel('Proportion', fontsize=16)
    plt.xlabel(r'Duration', fontsize=16)

    return df_disp, dfg


def illustrate_survival_curve_gene(df, gene_name, T, stbl=True,
                               show_HCT=True, randomize_HC=False,
                               show_stats_in_title=True):
    
    data_con = reduce_time_resolution(df.filter(['time', 'event', gene_name]), T=T)
    dfg = two_groups_table(data_con, gene_name)

    Nt1, Nt2 = dfg['at_risk:0'].values, dfg['at_risk:1'].values
    Ot1, Ot2 = dfg['observed:0'].values, dfg['observed:1'].values

    stats = evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, stbl=stbl, randomize=randomize_HC)
    stats_rev = evaluate_test_stats(Nt2, Nt1, Ot2, Ot1, stbl=stbl, randomize=randomize_HC)
    if stats['hc_greater'] < stats_rev['hc_greater']:  # reverse groups
        dfg = dfg.rename(columns={'at_risk:0': 'at_risk:1', 'at_risk:1': 'at_risk:0',
                                  'observed:0': 'observed:1', 'observed:1': 'observed:0',
                                  'censored:0': 'censored:1', 'censored:1': 'censored:0'
                                  })
        temp = stats
        stats = stats_rev
        stats_rev = temp

    Nt1, Nt2 = dfg['at_risk:0'].values, dfg['at_risk:1'].values
    Ot1, Ot2 = dfg['observed:0'].values, dfg['observed:1'].values

    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    pvals_rev = multi_pvals(Nt2, Nt1, Ot2, Ot1, randomize=False)
    fpval = find_changes(Nt1, Nt2, Ot1, Ot2, stbl=True)

    dfg['pvalue'] = pvals
    dfg['pvalue_rev'] = pvals_rev
    cumc1 = dfg['censored:0'].cumsum()
    cumc2 = dfg['censored:1'].cumsum()
    dfg['Survival Proportion X'] = (dfg['at_risk:0'] - dfg['censored:0']) / (dfg['at_risk:0'].max() - cumc1)
    dfg['Survival Proportion Y'] = (dfg['at_risk:1'] - dfg['censored:1']) / (dfg['at_risk:1'].max() - cumc2)
    # dfg['censored:0'] = dfg['at_risk:0'] - dfg['']

    df_disp = dfg[fpval].rename(columns={'at_risk:0': 'at_risk X', 'at_risk:1': 'at_risk Y',
                                         'observed:0': 'observed X', 'observed:1': 'observed Y'
                                         })

    plt.step(dfg.index, dfg['Survival Proportion X'], 'b', where='pre')
    plt.step(dfg.index, dfg['Survival Proportion Y'], 'r', where='pre')
    ct1 = dfg['censored:0'] > 0
    ct2 = dfg['censored:1'] > 0
    s1 = 10 * (dfg.loc[ct1, 'censored:0'].max() / dfg.loc[ct1, 'censored:0']).values
    s2 = 10 * (dfg.loc[ct2, 'censored:1'].max() / dfg.loc[ct2, 'censored:1']).values
    plt.scatter(dfg.index[ct1], dfg.loc[ct1, 'Survival Proportion X'],
                marker='+', c='b',
                s=s1, alpha=.5)
    plt.scatter(dfg.index[ct2], dfg.loc[ct2, 'Survival Proportion Y'],
                marker='+', c='r',
                s=s2, alpha=.5)

    plt.legend([r'$\hat{S}_x$', r'$\hat{S}_y$'], fontsize=16, loc=1)

    if show_HCT:
        plt.bar(dfg.index[:len(fpval)], fpval, color='k', alpha=.2, width=.5)


    if show_stats_in_title:
        stats_str = f": HC={np.round(stats['hc_greater'],2)}, HCrev={np.round(stats_rev['hc_greater'],2)}, Log-rank={np.round(stats['log_rank_greater'],2)}"
    plt.title(f"{gene_name}" + stats_str)
    plt.ylabel('proportion', fontsize=16)
    plt.xlabel(r'$t$ [Time]', fontsize=16)
    plt.ylim([0.7, 1.01])

    return df_disp, dfg


def main():
    parser = argparse.ArgumentParser(description='Illustrate Results')
    parser.add_argument('-data', type=str, help='SCANB gene expression data file',
                        default="Data/SCANB_groups_valid.csv")
    parser.add_argument('-T', type=int, help='number of intervals')
    parser.add_argument('-gene-names', nargs="*",
                         type=str, help='list of gene names')
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

        df_gene = df.filter([gene_name, 'time', 'event'])
        df_gene_T = reduce_time_resolution(df_gene, T=T)
        df_disp, dfg = illustrate_survival_curve_time2event(df_gene_T.rename(columns={gene_name : 'group'}))
        plt.ylim([0.7,1.01])

        logging.info(f"Writing survival curve figure to {fig_filename}.")
        plt.savefig(fig_filename)

        dfd = df_disp.copy()
        table_filename = outdir + f'{gene_name}.csv'
        logging.info(f"Writing table of suspected time instances to {table_filename}.")
        dfd.filter(['at_risk X', 'at_risk Y', 'observed X', 'observed Y', 'pvalue', 'pvalue_rev']) \
            .to_csv(table_filename)


if __name__ == '__main__':
    main()
