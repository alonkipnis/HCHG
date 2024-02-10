
"""
??? unclear what this script does
"""
from test_gene_expression import reduce_time_resolution, two_groups_table
from lifelines.statistics import logrank_test
import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] =  [8, 6]
mpl.style.use('ggplot')
import pandas as pd


weightings = [None, 'wilcoxon', 'tarone-ware', 'peto', 'fleming-harrington']

SELECTED_GENES = ['PBX1',
 'CLCF1',
 'MGAT4A',
 'SLC5A12',
 'DDX5',
 'MRAS',
 'IFNAR2',
 'RARRES1',
 'LDHB',
 'TLX2']

def logrank_lifeline_survival_table(df_table, **kwrgs):
    """
    Apply the logrank test from lifeline to survival table.
    To to so, we first need to convert the table to time-to-event representation
    by duplicating entries of removed subjects (observed or censored)
    """

    def table2time(df):
        df_obs = df.loc[df.index.repeat(df['observed'])]
        df_cen = df.loc[df.index.repeat(df['censored'])]

        df_obs['event'] = (df_obs['observed'] > 0) + 0.0
        df_cen['event'] = 0

        return pd.concat([df_obs, df_cen], axis=0).filter(['event'])

    dfg0 = df_table.filter(like=r':0')
    dfg1 = df_table.filter(like=r':1')

    dft0 = table2time(dfg0.rename(columns={'observed:0': 'observed', 'censored:0': 'censored'}))
    dft1 = table2time(dfg1.rename(columns={'observed:1': 'observed', 'censored:1': 'censored'}))

    return logrank_test(dft0.index, dft1.index,
                        event_observed_A=dft0['event'], event_observed_B=dft1['event'],
                        **kwrgs)


df = pd.read_csv("./Data/SCANB_groups_valid.csv")
gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]

div_probs = df.agg(['mean'])
thresh = 0.001
valid_genes = [g for g in gene_names if np.abs(div_probs[g]['mean'] - 0.5) < thresh]
invalid_genes = [g for g in gene_names if np.abs(div_probs[g]['mean'] - 0.5) > thresh]
df = df.drop(columns = invalid_genes + ['Unnamed: 0'])

assert(len(invalid_genes) == 0)

print("Removed: ", 9259 - len(valid_genes))


T = 512

wt = weightings[4]

for gene_name in SELECTED_GENES:

    df_gene = df.filter([gene_name, 'time', 'event']).rename(columns={gene_name: 'group'})
    df_gene_T = reduce_time_resolution(df_gene, T=T)

    ix = df_gene_T['group'] == 1
    T0, E0 = df_gene_T.loc[ix, 'time'], df_gene_T.loc[ix, 'event']
    T1, E1 = df_gene_T.loc[~ix, 'time'], df_gene_T.loc[~ix, 'event']

    results = logrank_test(T0, T1, event_observed_A=E0, event_observed_B=E1, weightings=wt, p=0, q=0)

    print("=========================")
    print(results.test_statistic)

    dfg = two_groups_table(df_gene_T, 'group')
    print(logrank_lifeline_survival_table(dfg, weightings=wt, p=0, q=0).test_statistic)
