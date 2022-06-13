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
    T = len(Nt1) - 1

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
        test_results['log_rank_greater'] = lr

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
        test_results['log_rank_less'] = lr

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


def run_one_experiment():
    T = 10000
    N1 = N2 = 1000
    beta = .7  # sparsity parameter
    eps = np.round(T ** -beta, 4)  # sparsity rate
    mu = .02  # signal strength

    Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, mu)

    res = evaluate_test_stats(Nt1, Nt2, randomized=True, alternative='both')
    print(res)


def simulate_null(N1, N2, T, nMonte, out_filename='under_null'):
    """
    Args:
    -----
    :N1:      Initial size of group 1
    :N2:      Initial size of group 2
    :T:       Total numebr of events
    :nMonte:  number of experiments to evaluate

    """
    fn = f"{out_filename}_T{T}_N{N1}.csv"  # output file

    df0 = pd.DataFrame()
    print("Simulating null...")
    for itr in tqdm(range(nMonte)):
        Nt1, Nt2 = sample_survival_data(T, N1, N2, 0, 0)
        res = evaluate_test_stats(Nt1, Nt2)
        df0 = df0.append(res, ignore_index=True)

    # critical values under the null:
    df0.agg([q95]).to_csv(fn)
    print("Results:")
    print(df0.agg([q95]))
    print(f"Stored results in {fn}")


def run_many_experiments(T, N1, N2):
    # under non-null
    bb = np.linspace(.5, .9, 7)
    rr = np.sqrt(np.linspace(0.01, 1, 9))
    N = np.min(N1, N2)
    mm = 2 * rr * np.log(T) / N

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


def evaluate():
    pass


def main():
    print("Check function `evalaute`...")
    r = evaluate()
    print(r)


if __name__ == '__main__':
    main()

# In[ ]:
