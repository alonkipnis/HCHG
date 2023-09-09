import numpy as np
import pandas as pd
import scipy

from twosample import binom_test_two_sided
from multitest import MultiTest
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')

from scipy.stats import poisson, norm, hypergeom, uniform
from sample_survival_poisson import sample_survival_poisson as sample_survival_data

import sys
sys.path.append("../")
from survival import evaluate_test_stats


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
    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)
    return evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2)


def run_one_experiment():
    T = 10000
    N1 = N2 = 1000
    beta = .7  # sparsity parameter
    eps = np.round(T ** -beta, 4)  # sparsity rate
    mu = .02  # signal strength

    Nt1, Nt2 = sample_survival_data(T, N1, N2, eps, mu)
    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)
    
    res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2, randomized=True, alternative='both')
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
        Ot1 = -np.diff(Nt1)
        Ot2 = -np.diff(Nt2)
        res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2)
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
    N = np.minimum(N1, N2)
    lam0 = np.ones(T)/T

    df1 = pd.DataFrame()
    nMonte = 100  # number of experiments

    for itr in tqdm(range(nMonte)):
        for beta in bb:
            for r in rr:
                eps = T ** -beta  # sparsity rate
                Nt1, Nt2 = sample_survival_data(T, N1, N2, lam0, eps, r)
                Ot1 = -np.diff(Nt1)
                Ot2 = -np.diff(Nt2)
                res1 = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2)
                res = pd.DataFrame(res1, index=[0])
                res['r'] = r
                res['eps'] = eps
                res['beta'] = beta
                res['itr'] = itr
                res['N1'] = N1
                res['N2'] = N2
                df1 = df1.append(res, ignore_index=True)
    return df1


def evaluate():
    r = run_many_experiments(100, 500, 500)
    print(r)


def main():
    print("Check function `evalaute`...")
    r = evaluate()
    print(r)


if __name__ == '__main__':
    main()

# In[ ]:
