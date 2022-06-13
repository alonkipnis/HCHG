import numpy as np
import pandas as pd
import scipy
import seaborn

from twosample import binom_test_two_sided
from multitest import MultiTest
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')

from scipy.stats import poisson, binom, norm, hypergeom, uniform
from sample_survival_data import *


def log_rank_test(Nt1, Nt2, alternative='two-sided'):
    """
    log-rank test 
    We assume that len(Nt1) == len(Nt2), and that each
    entry in either list represents an event in which
    a change occurs in the number of items in each groups 
    (the change in each group may also be zero)
    
    Args:
    -----
    :Nt1:   vector of counts in group 1 (each count corresponds to an event)
    :Nt2:   vector of counts in group 2 
    :alternative:   options are: 'greater', 'less', or 'two-sided'
                    with 'greater', test against the alternative that
                    more events occured in Nt2 compared to Nt1
    Returns:
    -------
    :z:       z score of the log-rank test
    :pvalue:  P-value
    """

    assert (len(Nt1) == len(Nt2))

    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)

    Nt = Nt2 + Nt1
    e0 = Nt2[:-1] * (Ot1 + Ot2) / Nt[:-1]
    var0 = e0 * ((Nt[:-1] - (Ot1 + Ot2)) / Nt[:-1]) * (Nt1[:-1] / (Nt[:-1] - 1))

    z = np.sum(Ot2 - e0) / np.sqrt(np.sum(var0))

    if alternative == 'greater':
        pval = norm.sf(z)
    elif alternative == 'less':
        pval = norm.cdf(z)
    else:
        pval = 2 * norm.cdf(-np.abs(z))

    return z, pval


def hypergeom_test(k, M, n, N, alternative='greater', randomize=False):
    """
    Exact hypergeometric test
    
    Args:
    -----
    :k:    number of observed Type I objects
    :M:    total number of object
    :n:    total number of Type I objects
    :N:    number of draws
    :randomize:   whether to do a randomized test
    :alternative: type of alternative to consider. Options are: 
                  'greater', 'less', 'two-sided'
    
    Returns:
        Test's P-value
    """

    if randomize:
        U = uniform.rvs(size=len(k))
    else:
        U = 0

    if alternative == 'greater':
        return hypergeom.sf(k, M, n, N) + U * hypergeom.pmf(k, M, n, N)
    if alternative == 'less':
        return hypergeom.cdf(k - 1, M, n, N) + U * hypergeom.pmf(k, M, n, N)
    raise ValueError("two-sided alternative is not available yet")


def q95(x):
    """
    Compute the 95-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, .95)


def mutli_pvals(Nt1, Nt2, test='hypergeom',
                randomize=True, alternative='greater'):
    """
    Compute P-values from the pair list of coutns in the two groups.
    We have one p-value per event time.
    An even is a pair (Nt1[i], Nt2[i]).
    
    Args:
    -----
    :Nt1:   vector of counts in group 1 (each count corresponds to an event)
    :Nt2:   vector of counts in group 2
    :test:  is the type of test to apply (options are: 'hypergeom' or
     'binomial')
    :randomize:  randomized individual tests or not
    :alternative:   type of alternative to use in each test 

    Return:
        P-values 
    """

    assert (len(Nt1) == len(Nt2))

    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)
    Nt = Nt2 + Nt1

    if test == 'binomial':
        n = Ot1 + Ot2
        p = Nt2[:-1] / Nt[:-1]
        x = Ot2
        pvals = binom_test_two_sided(x, n, p,
                                     randomize=randomize, alternative=alternative)
    elif test == 'hypergeom':
        pvals = hypergeom_test(Ot2, Nt[:-1], Nt2[:-1], Ot1 + Ot2,
                               randomize=randomize, alternative=alternative)

    return pvals


def atmoic_experiment(T, N1, N2, eps, mu):
    """
    Sample from survival data; evalaute several test statistics
    
    Args:
    -----
    :T:    time horizon (~ total number of events)
    :N1:   total in group1 at t=0
    :N2:   total in group1 at t=0
    :eps:  fraction of non-null events
    :mu:   intensity of non-null events
    
    """

    Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, mu)
    return evaluate_test_stats(Nt1, Nt2)


def evaluate_test_stats(Nt1, Nt2, **kwargs):
    """
    Args:
    :Nt1: first list of events
    :Nt2: second list of events
    
    Compute several statistics of the two-sample data:
    log-rank
    higher criticism
    Fisher combination test
    minimum P-value
    Berk-Jones
    Wilcoxon ranksum
    """

    randomize = kwargs.get('randomize', True)
    alternative = kwargs.get('alternative', 'both')  # 'both' != 'two-sided'

    test_results = {}

    if alternative == 'both' or alternative == 'greater':
        lr, lr_pval = log_rank_test(Nt1, Nt2, alternative='greater')
        test_results['log_rank_greater'] = -np.log(lr_pval)  # large values are significant

        pvals_greater = mutli_pvals(Nt1, Nt2, alternative='greater',
                                    randomize=randomize)
        mt = MultiTest(pvals_greater, stbl=False)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc_greater'] = mt.hc()[0]
        test_results['fisher_greater'] = mt.fisher()[0]
        test_results['min_p_greater'] = mt.minp()
        test_results['berk_jones_greater'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon_greater'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='greater').pvalue)

    if alternative == 'both' or alternative == 'less':
        lr, lr_pval = log_rank_test(Nt1, Nt2, alternative='less')
        test_results['log_rank_less'] = -np.log(lr_pval)  # large values are significant

        pvals_greater = mutli_pvals(Nt1, Nt2, alternative='less',
                                    randomize=randomize)
        mt = MultiTest(pvals_greater, stbl=False)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc_less'] = mt.hc()[0]
        test_results['fisher_less'] = mt.fisher()[0]
        test_results['min_p_less'] = mt.minp()
        test_results['berk_jones_less'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon_less'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='less').pvalue)

    if alternative == 'two-sided':
        lr, lr_pval = log_rank_test(Nt1, Nt2, alternative='two-sided')
        test_results['log_rank'] = lr

        pvals_greater = mutli_pvals(Nt1, Nt2, alternative='two-sided',
                                    randomize=randomize)
        mt = MultiTest(pvals_greater, stbl=False)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc'] = mt.hc()[0]
        test_results['fisher'] = mt.fisher()[0]
        test_results['min_p'] = mt.minp()
        test_results['berk_jones'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='two-sided').pvalue)

    return test_results


def simulate_null(N1, N2, T, nMonte):
    """
    Args:
    -----
    :N1:      Initial size of group 1
    :N2:      Initial size of group 2
    :T:       Total numebr of events
    :nMonte:  number of experiments to evaluate

    """

    df0 = pd.DataFrame()
    print("Simulating null...")
    for itr in tqdm(range(nMonte)):
        Nt1, Nt2 = sample_survival_data(T, N1, N2, 0, 0)
        res = evaluate_test_stats(Nt1, Nt2)
        df0 = df0.append(res, ignore_index=True)

    # critical values under the null:
    return df0.agg([q95])


def run_many_experiments(T, N1, N2, nMonte):
    # under non-null
    bb = np.linspace(.5, .9, 7)
    rr = np.sqrt(np.linspace(0.01, 1, 9))
    mm = 2 * rr * np.log(T) / N1

    df1 = pd.DataFrame()
    nMonte = 100  # number of experiments

    for itr in tqdm(range(nMonte)):
        for beta in bb:
            for mu in mm:
                eps = T ** -beta  # sparsity rate
                Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, mu)
                res1 = evaluate_test_stats(Nt1, Nt2)
                res2 = evaluate_test_stats(Nt2, Nt1)
                res = pd.DataFrame(res1, index=[0])
                res['mu'] = mu
                res['eps'] = eps
                res['beta'] = beta
                res['itr'] = itr
                df1 = df1.append(res, ignore_index=True)
    return df1


def illustrate_phase_diagrams(df1, df0, out_filename='phase_diagram'):
    """
    Args:
    -----
    :df1:   results from experiments under alternative 
    :df0:   results from experiments under null
    :out_filename:   file/path name to write to

    """

    params = ['itr', 'T', 'N1', 'N2' 'r', 'beta']
    tests = ['log_rank', 'hc', 'min_p', 'berk_jones', 'wilcoxon', 'fisher']
    tests_vars = [c for c in df0.columns if c not in params and 'Unnamed' not in c]

    global_params = ['T', 'N1', 'N2']
    tcrit = df0.groupby(global_params).agg(q95)

    good_side = 'greater'
    bad_side = 'less'

    for c in df1.groupby(global_params):
        print(f"Analyzing the case (T, N1, N2) = {c[0]}")
        dfc = c[1]

        for tsn in tests:
            name_good = tsn + '_' + good_side
            name_bad = tsn + '_' + bad_side

            # we check rate when both sides succedds. These are not good outcomes
            two_side_succ = (dfc[name_good] > tcrit[name_good].values[0]) & (dfc[name_bad] > tcrit[name_bad].values[0])
            print(f"{tsn}: both sides detected in {np.mean(two_side_succ)} of cases")
            print("(you should be worried if this number is significantly larger than 0.05)")

            bb = dfc['beta'].unique()
            rr = dfc['r'].unique()
            mat = np.zeros((len(bb), len(rr)))
            for i, beta in enumerate(bb):
                for j, r in enumerate(rr):
                    dfs = dfc[(dfc['beta'] == beta) & (dfc['r'] == r)]
                    succ = dfs[name_good] > tcrit[name_good].values[0]
                    mat[i, j] = np.mean(succ)

            plt.figure()
            g = seaborn.heatmap(mat[:, ::-1].T)
            plt.title(f"{tsn} (power at .05 level)")
            g.set_xticklabels(bb)
            g.set_xlabel('sparsity')
            g.set_ylabel('intensity')
            # g.set_yticklabels(np.round(mm[::-1],3))
            g.set_yticklabels(np.round(rr[::-1], 3))
            fn = out_filename + '_' + tsn + ".png"
            plt.savefig(fn)
            print("here")


def evaluate(itr, T, N1, N2, beta, r):
    """
    order of argument is important!
    evalaute an atomic experiment
    """
    mu = 2 * r * np.log(T) / np.log(T)
    eps = T ** (-beta)

    Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, mu)
    print(Nt1, Nt2)
    print(sum(Nt1 > 0))
    res = evaluate_test_stats(Nt1, Nt2, randomized=True, alternative='both')
    return res


def main():
    T = 1000
    N1 = 10000
    N2 = 10000
    beta = .7
    r = .2
    res = evaluate(1, T, N1, N2, beta, r)
    print(res)


if __name__ == '__main__':
    main()
