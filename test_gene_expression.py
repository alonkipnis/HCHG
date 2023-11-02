import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

from lifelines.utils import group_survival_table_from_events

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


def two_groups_table(data, group_indicator_name):
    """
    Create two-groups survival table from events
    
    at_risk columns is the number of individuals at risk at the beginning of the interval
    removed columns is the number of individuals removed from the risk set within the interval
    observed columns is the number of individuals that experienced the event within the interval
    censored columns is the number of individuals that were censored within the interval

    Args:
        :data: is a dataframe with columns: 'time', 'event', <group_indicator_name>

    Return:
        :dfr: is a dataframe with columns: at_risk:0, at_risk:1, removed:0, removed:1, observed:0, observed:1, censored:0, censored:1
        One entry per time interval
    """
    
    r = group_survival_table_from_events(groups=data[group_indicator_name], durations=data['time'], event_observed=data['event'])
    dfr = pd.concat(r[1:], axis=1)

    assert np.all(dfr['removed:0'] - dfr['censored:0'] == dfr['observed:0'])
    assert np.all(dfr['removed:1'] - dfr['censored:1'] == dfr['observed:1'])

    dfr = pd.concat(r[1:], axis=1)

    total0 = dfr['removed:0'].sum()
    a0 = dfr['removed:0'].cumsum()
    dfr['at_risk:0'] = (total0 - a0).shift(1).fillna(total0)

    total1 = dfr['removed:1'].sum()
    a1 = dfr['removed:1'].cumsum()
    dfr['at_risk:1'] = (total1 - a1).shift(1).fillna(total1)
    
    return dfr


def reduce_time_resolution(df, T=None, interval_duration=None):
    # reduce time resolution (over all genese simultenously)
    assert T is not None or interval_duration is not None, "At least one must be provided"

    tmax = df['time'].max()
    tmin = df['time'].min()

    if T is not None:
        interval_duration = (tmax - tmin) / T
    # otherwise use the provided interval_duration vlaue
    
    df_con = df.copy()
    df_con.loc[:, 'time'] = (df['time'] - tmin) // interval_duration
    return df_con



def test_gene(data, gene_name, T, stbl=True, randomize=False):
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
        data_con = reduce_time_resolution(data[['time', 'event', gene_name]], T=T)
        dfr = two_groups_table(data_con, gene_name)
    else:
        dfr = two_groups_table(data, gene_name)

    r = evaluate_test_stats(dfr['at_risk:0'].values, dfr['at_risk:1'].values,
                            dfr['observed:0'].values, dfr['observed:1'].values,
                            stbl=stbl, randomize=randomize, alternative='greater')
    rrev = evaluate_test_stats(dfr['at_risk:1'].values, dfr['at_risk:0'].values,
                            dfr['observed:1'].values, dfr['observed:0'].values,
                            stbl=stbl, randomize=randomize, alternative='greater')

    r['name'] = gene_name
    rrev['name'] = gene_name

    r['x0'] = dfr['at_risk:0'].max()
    r['y0'] = dfr['at_risk:1'].max()
    r['lam'] = (dfr['observed:0'].sum() + dfr['observed:1'].sum()) / (dfr['at_risk:0'].sum() + dfr['at_risk:1'].sum())
    
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
        :randomize:  whether to randomize P-values
        :rep:  number of repetitions in each sample. Useful for randomized statistics

    Return:
        :df0:  dataframe with test statistics as columns and genes as rows
               critical values for level alpha test can be obtained by
                from the 1-alpha quantile of each test statistic over all
                genes x repetitions
    """
    logging.info("Simulating null data by randomizing group assignments in the provided time-to-event data.")

    if randomize and (repetitions==1):
        logging.warning("Randomized statistics are not meaningful with one repetition. Setting repetitions to 100")
        repetitions = 100
    if not randomize and repetitions > 1:
        logging.warning("Non-randomized statistics are not meaningful with repetitions. Setting repetitions to 1")
        repetitions = 1

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
        df_test = pd.DataFrame({'random_sample': a + 0.0,
                                'time': df['time'],
                                'event': df['event']})
        df_test['random_sample'] = df_test['random_sample'].astype(int)
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

    #gene_names = pd.read_csv("genes_detected_by_HC.csv").iloc[:,1].tolist()

    logging.info(f"Testing {len(gene_names)} genes...")

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
    pd.DataFrame(res).to_csv(fn)

def main():
    parser = argparse.ArgumentParser(description='Analyze SCANB')
    parser.add_argument('-i', type=str, help='input file', default='./Data/SCANB_groups_valid.csv')
    parser.add_argument('-o', type=str, help='output file', default='SCANB')
    parser.add_argument('-T', type=int, help='number of instances', default=100)
    parser.add_argument('-M', type=int, help='repetitions', default=1)
    parser.add_argument('-nMonte', type=int, help='number of Monte-Carlo repetitions for null evalautions', default=10000)
    

    parser.add_argument('--null', action='store_true', help='simulate null data (random group assignments)')
    parser.add_argument('--stbl', action='store_true', help='type of HC denumonator')
    parser.add_argument('--randomize', action='store_true', help='randomized hypergeometric P-values')
    args = parser.parse_args()
    #

    T = args.T
    stbl = args.stbl
    print("stbl = ", stbl)

    print(f"Reading data from {args.i}...")
    df = pd.read_csv(args.i)
    
    if args.null:
        print("Simulating null...")
        res = simulate_null_data(df, T, stbl=stbl, repetitions=args.M, randomize = args.randomize, nMonte=args.nMonte)
        rand_str = "_randomized" if args.randomize else ""
        stbl_str = "stable" if stbl else "not_stable"
        fn = f'{args.o}_null_{stbl}_T{T}{rand_str}_rep{args.M}.csv'
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
