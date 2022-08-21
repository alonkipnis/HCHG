import numpy as np
import pandas as pd
import scipy
import seaborn

from twosample import binom_test
from multitest import MultiTest
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')

from scipy.stats import poisson, norm, hypergeom, uniform
from sample_survival_data import *
from lifelines.statistics import logrank_test as logrank_lifeline

EPS = 1e-20


def log_rank_test(Nt1, Nt2, Ot1, Ot2, alternative='two-sided'):
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
    assert (len(Ot1) == len(Ot2))
    assert (len(Ot1) == len(Nt1))

    Nt = Nt2 + Nt1
    e0 = Nt2 * (Ot1 + Ot2) / Nt
    var0 = e0 * ((Nt - (Ot1 + Ot2)) / Nt) * (Nt1 / (Nt - 1))

    z = np.sum(Ot2 - e0) / np.sqrt(np.sum(var0))

    if alternative == 'greater':
        pval = norm.sf(z)
    elif alternative == 'less':
        pval = norm.cdf(z)
    else:
        pval = 2 * norm.cdf(-np.abs(z))

    return z, pval


def log_rank_test_lifeline(Nt1, Nt2):
    lr = logrank_lifeline(Nt1, Nt2).summary
    return lr['test_statistic'][0], lr['p'][0]


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
        U = 1

    if alternative == 'greater':
        return hypergeom.sf(k, M, n, N) + U * hypergeom.pmf(k, M, n, N)
    if alternative == 'less':
        return hypergeom.cdf(k - 1, M, n, N) + (1-U) * hypergeom.pmf(k, M, n, N)
    raise ValueError("two-sided alternative is not available yet")


def q95(x):
    """
    Compute the 95-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, .95)


def multi_pvals(Nt1, Nt2, Ot1, Ot2, test='hypergeom',
                randomize=False, alternative='greater'):
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
    assert (len(Ot1) == len(Ot2))
    assert (len(Ot1) == len(Nt1))

    Nt = Nt2 + Nt1

    if test == 'binomial':
        n = Ot1 + Ot2
        p = Nt2 / Nt
        x = Ot2
        pvals = binom_test(x, n, p, randomize=randomize, alt=alternative)
    elif test == 'hypergeom':
        pvals = hypergeom_test(Ot2, Nt, Nt2, Ot1 + Ot2,
                               randomize=randomize, alternative=alternative)

    return pvals


def atmoic_experiment(T, N1, N2, eps, r):
    """
    Sample from survival data; evalaute several test statistics
    
    Args:
    -----
    :T:    time horizon (~ total number of events)
    :N1:   total in group1 at t=0
    :N2:   total in group1 at t=0
    :eps:  fraction of non-null events
    :r:   intensity of non-null events
    
    """

    Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, r)
    return evaluate_test_stats(Nt1, Nt2)


def evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, **kwargs):
    """
    Evaluate many tests for comparing the lists Nt1 and Nt2

    Args:
    :Nt1: first list of at-risk subjects
    :Nt2: second list of at-risk subjects
    :Ot1: number of events in group 1
    :Ot2: number of events in group 2


    Compute several statistics of the two-sample data:
    log-rank
    higher criticism
    Fisher combination test
    minimum P-value
    Berk-Jones
    Wilcoxon ranksum
    """

    randomize = kwargs.get('randomize', False)
    alternative = kwargs.get('alternative', 'both')  # 'both' != 'two-sided'
    stbl = kwargs.get('stbl', True)
    discard_ones = kwargs.get('discard_ones', True) # ignore P-values that are one

    test_results = {}

    #test_results['log_rank_lifeline'] = -np.log(lrln_pval + EPS)

    if alternative == 'both' or alternative == 'greater':
        lr, lr_pval = log_rank_test(Nt1, Nt2, Ot1, Ot2, alternative='greater')
        test_results['log_rank_greater'] = np.abs(lr)  # large values are significant

        pvals_greater = multi_pvals(Nt1, Nt2, Ot1, Ot2, alternative='greater',
                                    randomize=randomize)
        if discard_ones:
            pvals_greater = pvals_greater[pvals_greater < 1]
        mt = MultiTest(pvals_greater, stbl=stbl)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc_greater'] = mt.hc()[0]
        test_results['fisher_greater'] = mt.fisher()
        test_results['min_p_greater'] = mt.minp()
        test_results['berk_jones_greater'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon_greater'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='greater').pvalue + EPS)

    if alternative == 'both' or alternative == 'less':
        lr, lr_pval = log_rank_test(Nt1, Nt2, Ot1, Ot2, alternative='less')
        test_results['log_rank_less'] = np.abs(lr)  # large values are significant

        pvals_greater = multi_pvals(Nt1, Nt2, Ot1, Ot2, alternative='less',
                                    randomize=randomize)
        if discard_ones:
            pvals_greater = pvals_greater[pvals_greater < 1]
        mt = MultiTest(pvals_greater, stbl=False)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc_less'] = mt.hc()[0]
        test_results['fisher_less'] = mt.fisher()
        test_results['min_p_less'] = mt.minp()
        test_results['berk_jones_less'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon_less'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='less').pvalue + EPS)

    if alternative == 'two-sided':
        lr, lr_pval = log_rank_test(Nt1, Nt2, Ot1, Ot2, alternative='two-sided')
        test_results['log_rank'] = np.abs(lr)

        pvals_greater = mutli_pvals(Nt1, Nt2, Ot1, Ot2, alternative='two-sided',
                                    randomize=randomize)
        if discard_ones:
            pvals_greater = pvals_greater[pvals_greater < 1]
        mt = MultiTest(pvals_greater, stbl=False)
        # if not using stbl=False, then sometimes
        # HC misses the significance of the strongest effect
        test_results['hc'] = mt.hc()[0]
        test_results['fisher'] = mt.fisher()
        test_results['min_p'] = mt.minp()
        test_results['berk_jones'] = mt.berk_jones(gamma=.45)
        test_results['wilcoxon'] = -np.log(scipy.stats.ranksums(
            Nt1, Nt2, alternative='two-sided').pvalue + EPS)

    return test_results


def simulate_null(T, N1, N2, lam0, nMonte):
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
        Nt1, Nt2 = sample_survival_data(T, N1, N2, lam0, 0, 0)
        Ot1 = -np.diff(Nt1)
        Ot2 = -np.diff(Nt2)
        res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2, stbl=True)
        df0 = df0.append(res, ignore_index=True)

    # critical values under the null:
    return df0


def run_many_experiments(T, N1, N2, lam0, nMonte):
    # under non-null
    bb = np.linspace(.5, .9, 7)
    rr = np.sqrt(np.linspace(0.01, 1, 9))

    df1 = pd.DataFrame()

    for itr in tqdm(range(nMonte)):
        for beta in bb:
            for r in rr:
                eps = T ** -beta  # sparsity rate
                Nt1, Nt2 = sample_survival_data(T, N1, N2, lam0, eps, r)
                Ot1 = -np.diff(Nt1)
                Ot2 = -np.diff(Nt2)
                res1 = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2, stbl=True)
                res = pd.DataFrame(res1, index=[0])
                res['mu'] = mu
                res['eps'] = eps
                res['beta'] = beta
                res['itr'] = itr
                res['lam0'] = lam0
                df1 = df1.append(res, ignore_index=True)
    return df1


def evaluate(itr, T, N1, N2, lam0, beta, r):
    """
    order of argument is important!
    evalaute an atomic experiment
    """

    eps = T ** (-beta)
    lam = lam0 * np.ones(T) / T

    Nt1, Nt2 = sample_survival_data(T, N1, N2, lam, eps, r)
    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)
    res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2,
                              randomized=True, alternative='both', stbl=True)
    return res


def main():
    T = 1000
    N1 = 5000
    N2 = 5000
    beta = .7
    r = 1
    lam0 = 3

    res = evaluate(1, T, N1, N2, lam0, beta, r)
    print(res)


if __name__ == '__main__':
    main()
