import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')
from tqdm import tqdm
import pandas as pd
import argparse

import logging


logging.basicConfig(level=logging.INFO)
from survival import (q95, evaluate_test_stats)


def infmean(x):
    "mean ignoring inf values"
    return np.mean(np.ma.masked_invalid(x))


def std_95(x):
    return scipy.stats.mstats.mjci(x, prob = [0.95])[0]



def infstd(x):
    return np.std(np.ma.masked_invalid(x))


def arrange_group(dfg):
    """
    Group events in time; Find group size in each
    time sample
    """

    dft = dfg  # .groupby('time').sum()
    dft = dft.sort_values('time').reset_index()
    dft['total'] = len(dft)
    dft['dead'] = dft.event
    dft['cum_dead'] = dft.dead.cumsum()

    # dft['censored'] = (~dft.event).cumsum()
    # incorporate censorship. Change to 'at-risk'
    dft['cum_censored'] = (dft.event == 0).cumsum()
    dft['censored'] = (dft.event == 0) + 0.0
    dft['at-risk'] = dft['total']
    dft.loc[1:, 'at-risk'] = dft['total'].max() - dft[:-1]['cum_dead'].values - dft['cum_censored'].values[1:]
    dft = dft.groupby('time').max()
    assert (np.all(-np.diff(dft['at-risk']) >= dft['dead'].values[:-1]))
    return dft


def two_groups_gene(data, gene_name):
    """
    Arranges relevant data in two groups format for survival analysis based on
    the examined gene

    Args:
    :data:   is a dataframe with :gene_name: as one of its columns.
             additional columns are :time: and :event:

    Return:
        dataframe indexed by time and number of survived elements in
        each group

    """

    dfg = data[[gene_name, 'time', 'event']]
    idc_split = dfg[gene_name] >= 1
    df1 = arrange_group(dfg[idc_split])
    df2 = arrange_group(dfg[~idc_split])

    dfm = df1[['at-risk', 'dead', 'censored', 'total']].join(df2[['at-risk', 'dead', 'censored', 'total']], lsuffix='1',
                                                             rsuffix='2', how='outer')
    dfm['dead1'] = dfm['dead1'].fillna(0)
    dfm['dead2'] = dfm['dead2'].fillna(0)
    dfm['censored1'] = dfm['censored1'].fillna(0)
    dfm['censored2'] = dfm['censored2'].fillna(0)
    dfm['total1'] = dfm['total1'].fillna(method='bfill').fillna(method='ffill')
    dfm['total2'] = dfm['total2'].fillna(method='bfill').fillna(method='ffill')

    dfm['at-risk1'] = dfm['total1']
    dfm['at-risk2'] = dfm['total2']
    dfm.loc[dfm.index[1:], 'at-risk1'] = dfm['total1'].values[0] \
                                         - dfm['dead1'].cumsum().values[:-1] - dfm['censored1'].cumsum().values[:-1]
    dfm.loc[dfm.index[1:], 'at-risk2'] = dfm['total2'].values[0] \
                                         - dfm['dead2'].cumsum().values[:-1] - dfm['censored2'].cumsum().values[:-1]

    return dfm


def reduce_time_resolution(df, T):
    """
    Group together events in surivial data
    across uniform time intervals.

    Args:
    :df: original dataset. Index represent time of events
    :T:  maximal number of time intervals

    """

    Tmin = df.index.min()
    Tmax = df.index.max()
    tt = np.linspace(Tmin, Tmax, T + 1)
    dfc = pd.DataFrame()
    for t_down, t_up in zip(tt[:-1], tt[1:]):
        dft = df[(t_down <= df.index) & (df.index < t_up)]
        r = dft.sum()[['dead1', 'dead2', 'censored1', 'censored2']]
        r['at-risk1'] = dft['at-risk1'].max()
        r['at-risk2'] = dft['at-risk2'].max()
        r_df = pd.DataFrame(r).T
        dfc = pd.concat([dfc, r_df])
    dfc = dfc.fillna(method='backfill').dropna()
    dfc['t'] = np.arange(0, T)
    return dfc.set_index('t')


def test_gene(data, gene_name, T, stbl=False, randomize=False):
    """
    Evaluate all test statistics for comparing the survival curves
    for the two groups involving the response of :gene_name:
    Also, consolidate event times based on :T:.

    Args:
    :data: is the entire dataset containing the group division
            at minimum, :data: should contain the columns
            :gene_name:, `events`, `time`
    :gene_name:  the name of the column to test
    :T:    is the new maximal time interval. If T == -1 then we
            use the original times
    :stbl:  type of denumerator in HC statisitc
    :randomize:  whether to randomize the Hypergeometric tests or not
    :repetitions:  number of times to randomize the test

    """
    if T > 0:
        dfr = reduce_time_resolution(two_groups_gene(data, gene_name), T)
    else:
        dfr = two_groups_gene(data, gene_name)

    r = evaluate_test_stats(dfr['at-risk1'].values, dfr['at-risk2'].values,
                            dfr['dead1'].values, dfr['dead2'].values,
                            stbl=stbl, randomize=randomize, alternative='greater')
    rrev = evaluate_test_stats(dfr['at-risk2'].values, dfr['at-risk1'].values,
                            dfr['dead2'].values, dfr['dead1'].values,
                            stbl=stbl, randomize=randomize, alternative='greater')

    r['name'] = gene_name
    rrev['name'] = gene_name

    r['x0'] = dfr['at-risk1'].max()
    r['y0'] = dfr['at-risk2'].max()
    r['lam'] = (dfr['dead1'].sum() + dfr['dead2'].sum()) / (dfr['at-risk1'].sum() + dfr['at-risk2'].sum())
    
    rdf = pd.DataFrame(r, index=[0])
    revdf = pd.DataFrame(rrev, index=[0])

    return rdf.join(revdf, rsuffix='_rev')


def simulate_null_data(df, T, stbl=True, repetitions=1, randomize=False, nMonte=10000):
    """
    Generate random partition and Evaluate test statistics. 

    Here we assume that df
    is the data in which the group assignments are
    in columns 1 to -2 (everything excluding the first column and last two)

    Args:
        :df:   data in a dataframe format
        :T:    number of time instances to consolidate the data to
        :stbl: parameter for type of HC to use
        :randimize:  whether to randomize P-values
        :rep:  number of repetitions

    Return:
        :df0:  dataframe with test statistics as columns and genes as rows
               critical values for level alpha test can be obtained by
                from the 1-alpha quantile of each test statistic over all
                genes x repetitions
    """

    def sample_balanced_assignmet(T):
        """ Perfectly balanced assignment """
        a = np.random.rand(T)
        return a < np.median(a)

    df0 = pd.DataFrame()
    logging.debug("Simulating null using real data...")

    df_test = df[['time', 'event']]

    for itr in tqdm(range(nMonte)):
        logging.debug(f'Sampling a random assignment')
        a = sample_balanced_assignmet(len(df_test))
        df_test = pd.DataFrame({'random_sample': a,
                                'time': df['time'],
                                'event': df['event']})
        # df_test.loc[:, 'random_sample'] = a
        res_df = pd.DataFrame()
        for _ in range(repetitions):
            r = test_gene(df_test, 'random_sample', T, stbl=stbl, randomize=randomize)
            res_df = pd.concat([res_df, r], axis=0)
        res_df['itr'] = itr
        df0 = pd.concat([df0, res_df])
        df0.to_csv("temp.csv")
    return df0


def main_test_all_genes(df, T, stbl=False, repetitions=1, randomize=False):
    gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]
    logging.info("Testing all genes...")

    gene_names = pd.read_csv("genes_detected_by_HC.csv").iloc[:,1].tolist()

    print(gene_names)
    print(f"Testing {len(gene_names)} genes...")

    res = pd.DataFrame()
    for gene_name in tqdm(gene_names):
        for _ in range(repetitions):
            r = test_gene(df, gene_name, T, stbl=stbl, randomize=randomize)
            res = pd.concat([res, r], axis=0)
    return res


def report_results(df0, res):
    crit_vals = df0.agg([q95]).filter(
        ['log_rank_greater', 'hc_greater', 'hc_greater_rev', 'log_rank_greater_rev'])

    LRt = crit_vals.loc['q95', 'log_rank_greater']
    HCt = crit_vals.loc['q95', 'hc_greater']
    HCp = crit_vals.loc['q95', 'hc_greater']

    log_rank_0 = (res.log_rank_greater <= LRt) & (res.log_rank_greater_rev <= LRt)
    log_rank_1 = (res.log_rank_greater > LRt) | (res.log_rank_greater_rev > LRt)

    hc_0 = (res.hc_greater <= HCt) & (res.hc_greater_rev <= HCp)
    hc_1 = ((res.hc_greater > HCt) & (res.hc_greater_rev < HCp)) | ((res.hc_greater < HCp) & (res.hc_greater_rev > HCt))

    HC1_LR1 = np.sum(log_rank_1 & hc_1)
    HC1_LR0 = np.sum(log_rank_0 & hc_1)
    HC0_LR1 = np.sum(log_rank_1 & hc_0)
    HC0_LR0 = np.sum(log_rank_0 & hc_0)

    print("Discoverable by HC: ", np.sum(hc_1))
    print("Discoverable by LR: ", np.sum(log_rank_1))
    print("Discoverable by HC and LR: ", HC1_LR1)
    print("Discoverable by HC but not LR: ", HC1_LR0)
    print("Discoverable by LR but not HC: ", HC0_LR1)
    print("Discoverable by neither LR nor HC: ", HC0_LR0)
    print("Total reported: ", HC1_LR1 + HC1_LR0 + HC0_LR1 + HC0_LR0)
    print("Total: ", len(res))

    return res[log_rank_0 & hc_1]


def save_results(res, fn):
    print(f"Saving to {fn}")
    #import pdb; pdb.set_trace()
    pd.DataFrame(res).to_csv(fn)

def main():
    parser = argparse.ArgumentParser(description='Analyze SCANB')
    parser.add_argument('-i', type=str, help='input file', default='./Data/SCANB_groups.csv')
    parser.add_argument('-o', type=str, help='output file', default='SCANB')
    parser.add_argument('-T', type=int, help='number of instances', default=100)
    parser.add_argument('-M', type=int, help='repetitions', default=1)
    

    parser.add_argument('--null', action='store_true', help='simulate null data (random group assignments)')
    parser.add_argument('--stbl', action='store_true', help='type of HC denumonator')
    parser.add_argument('--randomize', action='store_true', help='randomized hypergeometric P-values')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--report-null-stats', action='store_true')
    args = parser.parse_args()
    #

    T = args.T
    if args.report_null_stats:
        df0 = pd.read_csv(args.i)
        report_null_stats(df0, T)

    stbl = args.stbl
    M =args.M
    print("stbl = ", stbl)

    print(f"Reading data from {args.i}...")
    df = pd.read_csv(args.i)
    
    if args.null:
        print("Simulating null...")
        res = simulate_null_data(df, T, stbl=stbl, repetitions=args.M, randomize = args.randomize)
        fn = f'{args.o}_null_{stbl}_T{T}_{args.randomize}_rep{args.M}.csv'
        save_results(res, fn)
    elif args.randomize:
        res = main_test_all_genes(df, T, stbl, repetitions=args.M, randomize=True)
        fn = f'{args.o}_{stbl}_T{T}_randomized_rep{args.M}.csv'
        save_results(res, fn)
    else:
        res = main_test_all_genes(df, T, stbl)
        fn = f'{args.o}_{stbl}_T{T}.csv'
        save_results(res, fn)


if __name__ == '__main__':
    main()
