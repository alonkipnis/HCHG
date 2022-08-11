from twosample import binom_test
from multitest import MultiTest
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] =  [8, 6]
mpl.style.use('ggplot')
from tqdm import tqdm
from scipy.stats import poisson, binom, norm, hypergeom, uniform
import pandas as pd
from sample_survival_data import *
import argparse

from survival import ( hypergeom_test, q95, log_rank_test, multi_pvals, evaluate_test_stats)


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
    dft['at-risk'] = dft['total']
    dft.loc[1:, 'at-risk'] = dft[1:]['at-risk'].values - dft[:-1]['cum_dead'].values - dft[:-1]['cum_censored'].values
    dft = dft.groupby('time').max()
    assert (np.all(-np.diff(dft['at-risk']) >= dft['dead'].values[:-1]))
    return dft


def two_groups_gene(data, gene_name):
    """
    Arranges relevant data in a two groups format
    for survival analysis based on the examined gene

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

    dfm = df1[['at-risk', 'dead']].join(df2[['at-risk', 'dead']], lsuffix='1', rsuffix='2', how='outer') \
        .fillna(method='ffill') \
        .fillna(method='bfill')
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
    tt = np.linspace(Tmin, Tmax, T)
    dfc = pd.DataFrame()
    for t_down, t_up in zip(tt[:-1], tt[1:]):
        dft = df[(t_down <= df.index) & (df.index < t_up)]
        r = dft.sum()[['dead1', 'dead2']]
        r['at-risk1'] = dft['at-risk1'].max()
        r['at-risk2'] = dft['at-risk2'].max()
        dfc = dfc.append(r, ignore_index=True)
    return dfc.fillna(method='backfill').dropna()


def test_gene(data, gene_name, T, stbl=False):
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

    """
    if T > 0:
        dfr = reduce_time_resolution(two_groups_gene(data, gene_name), T)
    else:
        dfr = two_groups_gene(df, gene_name)
    r = evaluate_test_stats(dfr['at-risk1'].values, dfr['at-risk2'].values,
                            dfr['dead1'].values, dfr['dead2'].values,
                            stbl=stbl, randomize=False
                            )
    rrev = evaluate_test_stats(dfr['at-risk2'].values, dfr['at-risk1'].values,
                               dfr['dead2'].values, dfr['dead1'].values,
                               stbl=stbl, randomize=False)

    r['name'] = gene_name
    rrev['name'] = gene_name

    r['x0'] = dfr['at-risk1'].max()
    r['y0'] = dfr['at-risk2'].max()
    r['lam1'] = ( np.log(dfr['at-risk1'].max()) - np.log(dfr['at-risk1'].min()) ) / T
    r['lam2'] = ( np.log(dfr['at-risk2'].max()) - np.log(dfr['at-risk2'].min()) ) / T

    rdf = pd.DataFrame(r, index=[0])
    revdf = pd.DataFrame(rrev, index=[0])

    return rdf.join(revdf, rsuffix='_rev')


def std_95(x):
    """
    Standard error in estimating the 95 quantile based on the vector of
    measurements x.
    """
    return np.std(x) / np.sqrt(len(x)) * np.sqrt(.95 * (1 - .95))


def simulate_null_data(df, T, rep=1, stbl=False):
    """
    Generate random group assignments.
    Evaluate test statistics. Here we assume that df
    is the data in which the group assignments are
    in columns 1 to -2 (everything excluding the first column and last two)

    Args:
        :df:   data in a dataframe format
        :T:    number of time instances to consolidate the data to
        :rep:  number of random assignments
        :stbl: parameter for type of HC to use

    Return:
        :df0:  dataframe with test statistics as columns and genes as rows
               critical values for level alpha test can be obtained by
                from the 1-alpha quantile of each test statistic over all
                genes x repetitions
    """

    df0 = pd.DataFrame()
    gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]

    print("Simulating null using data...")

    for _ in range(rep):
        print("Sampling a random division...")
        df_copy = df.copy()
        df_copy.iloc[:, 1:-2] = np.random.randint(2, size=df.iloc[:, 1:-2].shape)
        print(f"Testing {len(gene_names)} 'null' genes...")
        for gene_name in tqdm(gene_names):
            res = test_gene(df_copy, gene_name, T, stbl=stbl)
            df0 = df0.append(res, ignore_index=True)

    return df0



def main_test_all_genes(df, T=100, stbl=False):
    gene_names = [c for c in df.columns if c not in ['Unnamed: 0', 'time', 'event']]

    print("Testing all genes...")

    res = pd.DataFrame()
    for gene_name in tqdm(gene_names[:100]):
        r = test_gene(df, gene_name, T, stbl=stbl)
        res = res.append(r, ignore_index=True)

    return res


def main():
    parser = argparse.ArgumentParser(description='Analyze SCNAB')
    parser.add_argument('-i', type=str, help='input file', default='./Data/SCNAB_groups.csv')
    parser.add_argument('-o', type=str, help='output file', default='SCNAB')
    parser.add_argument('-T', type=int, help='number of instances', default=100)

    parser.add_argument('--null', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    #

    T = args.T
    stbl = False

    print(f"Reading data from {args.i}...")
    df = pd.read_csv(args.i)

    if args.null:
        print("Simulating null...")
        res = simulate_null_data(df, T, rep=1, stbl=False)
        print(res.agg([q95, 'mean', 'std']))
        fn = f'{args.o}_null_{stbl}_T{T}.csv'
    elif args.test:
        res = main_test_all_genes(df, T, stbl)
        fn = f'{args.o}_{stbl}_T{T}.csv'

    print(f"Saving to {fn}")
    res.to_csv(fn)





#np.seterr(all='warn')





if __name__ == '__main__':
    main()