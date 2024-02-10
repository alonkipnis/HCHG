import pandas as pd
from multitest import MultiTest
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from multiHGtest import hypergeom_test
from scipy.stats import norm

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')

from phase_transition_experiment.sample_survival_poisson import *
from lifelines.statistics import logrank_test

from test_konp import evaluate_test_stats_konp

STBL = True
EPS = 1e-20


#from rpy2.robjects.packages import importr
#konp_test = importr('KONPsurv')

# def konp_testR(times, status, groups):
#     """
#     Apply the KONP test from R to survival table. (https://cran.r-project.org/web/packages/KONPsurv/)

#     """
#     res = konp_test.konp_test(times, status, groups, n_perm=1)
#     return dict(chisq_test_stat=res[3][0], lr_test_stat=res[4][0],cauchy_test_stat=res[5][0])


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
    valid_idcs = (Nt1 != 0) & (Nt2 != 0)

    Nt1 = Nt1[valid_idcs]
    Nt2 = Nt2[valid_idcs]
    Ot1 = Ot1[valid_idcs]
    Ot2 = Ot2[valid_idcs]

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

    # alternative only affecting pval, not z score
    return z, pval


def q95(x):
    """
    The 95-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, .95)


def time2event(times, num_events, num_censored, initial_at_risk):
    """
    Convert number of events and censored to time-to-event data.

    The number of events and censored at each time point is converted to a list of event times and event status.
    All events are assumed to occur at the same time as the time point.
    Events are coded as 1 and censored as 0.
    All at risk at the last time point are assumed to be censored.
    """

    assert len(times) == len(num_events) == len(num_censored) 
    assert initial_at_risk >= np.sum(num_events) + np.sum(num_censored)

    num_censored[-1] = num_censored[-1] + initial_at_risk - np.sum(num_events) - np.sum(num_censored)
    r_times = []
    r_status = []
    for i,t in enumerate(times):
        ne = num_events[i]
        nc = num_censored[i]
        r_times += ([t] * (ne + nc))
        r_status += ([1] * ne + [0] * nc)
    
    return r_times, r_status


def logrank_lifeline_survival_table(df_table, **kwrgs):
    """
    Apply the logrank test from lifeline to survival table.
    To do so, we first need to convert the table to time-to-event representation
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


def multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False, alternative='greater'):
    """
    Compute P-values from the pair list of coutns in the two groups.
    We have one p-value per event time.
    An even is a pair (Nt1[i], Nt2[i]).

    Args:
    -----
    :Nt1:   vector of counts in group 1 (each count corresponds to an event)
    :Nt2:   vector of counts in group 2
    :randomize:  randomized individual tests or not
    :alternative:   type of alternative to use in each test

    Return:
        P-values
    """

    assert (len(Nt1) == len(Nt2))
    assert (len(Ot1) == len(Ot2))
    assert (len(Ot1) == len(Nt1))

    pvals = hypergeom_test(Ot2, Nt2 + Nt1, Nt2, Ot1 + Ot2,
                            randomize=randomize, alternative=alternative)
    return pvals



def evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, **kwargs):
    """
    Evaluate many tests for comparing the lists Nt1 and Nt2

    Args:
    :Nt1: first list of at_risk subjects
    :Nt2: second list of at_risk subjects
    :Ot1: number of events in group 1
    :Ot2: number of events in group 2


    Compute several statistics of the two-sample data:
    log-rank
    higher criticism
    Fisher combination test
    minimum P-value
    Berk-Jones
    """

    randomize = kwargs.get('randomize', False)
    alternative = kwargs.get('alternative', 'both')  # NOTE: 'both' != 'two-sided'
    stbl = kwargs.get('stbl', True)
    discard_ones = kwargs.get('discard_ones', False)  # ignore P-values that are exactly one
    no_censoring = kwargs.get('no_censoring', False) # for data integrity

    if alternative == 'both':
        r_greater = _evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, alternative='greater',
                                         stbl=stbl, randomize=randomize, 
                                         discard_ones=discard_ones)
        r_less = _evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, alternative='less',
                                      stbl=stbl, randomize=randomize,
                                        discard_ones=discard_ones)

        res = dict([(k + '_greater', r_greater[k]) for k in r_greater.keys()]
                    + [(k + '_less', r_less[k]) for k in r_less.keys()]
                    )
    else:
        r = _evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, alternative=alternative,
                                 stbl=stbl, randomize=randomize, discard_ones=discard_ones)
        res = dict([(k + '_' + alternative, r[k]) for k in r.keys()])


    N1 = Nt1[0]
    N2 = Nt2[0]
    Nt1 = np.concatenate([Nt1, [Nt1[-1]-Ot1[-1]]], axis=0)
    Nt2 = np.concatenate([Nt2, [Nt2[-1]-Ot2[-1]]], axis=0)
    Ct1 = (-np.diff(Nt1) - Ot1).astype(int)
    Ct2 = (-np.diff(Nt2) - Ot2).astype(int)

    assert np.all(Ct1 >= 0)
    assert np.all(Ct2 >= 0)
    
    if no_censoring:
        assert np.abs(Ct1).sum() == 0
        assert np.abs(Ct2).sum() == 0
    

    Ct1[-1] = Ct1[-1] + N1 - Ot1.sum()
    Ct2[-1] = Ct2[-1] + N2 - Ot2.sum()
    res_ll = evaluate_test_stats_lifeline(Ot1, Ot2, Ct1, Ct2)
    res_konp = evaluate_test_stats_konp(Ot1, Ot2, Ct1, Ct2)

    return {**res, **res_ll, **res_konp}



def _evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, alternative,
                         randomize=False,
                         stbl=True, discard_ones=False):
    """
    Evaluate many tests for comparing the lists Nt1 and Nt2

    Args:
    :Nt1: first list of at_risk subjects
    :Nt2: second list of at_risk subjects
    :Ot1: number of events in group 1
    :Ot2: number of events in group 2


    Compute several statistics of the two-sample data:
    log-rank
    higher criticism
    Fisher combination test
    minimum P-value
    Berk-Jones
    """

    test_results = {}
    lr, lr_pval = log_rank_test(Nt1, Nt2, Ot1, Ot2, alternative=alternative)
    test_results['log_rank'] = lr  # large values are significant
    test_results['log_rank_pval'] = lr_pval  # alternative only affecting p-value, not lr score

    hcv = []
    fisherv = []
    minpv = []
    bjv = []
    
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, alternative=alternative,
                        randomize=randomize)
    if discard_ones:
        pvals = pvals[pvals < 1]
    mt = MultiTest(pvals, stbl=stbl)
    # if not using stbl=False, then sometimes
    # HC cannot detect a single dominant effect

    hcv = mt.hc(gamma=0.2)[0]
    fisherv = mt.fisher()[0]
    minpv = mt.minp()
    bjv = mt.berk_jones(gamma=.45)
    
    test_results['hc'] = hcv
    test_results['fisher'] = fisherv
    test_results['min_p'] = minpv
    test_results['berk_jones'] = bjv
    
    return test_results


def evaluate_test_stats_lifeline(Ot1, Ot2, Ct1, Ct2):
    res = {}
    dfg = pd.DataFrame({'observed:0' : Ot1, 'observed:1': Ot2, 'censored:0': Ct1, 'censored:1': Ct2})
    weightings = [None, 'wilcoxon', 'tarone-ware', 'peto', 'fleming-harrington55', 'fleming-harrington11', 'fleming-harrington01']
    for wt in weightings:
        if wt == 'fleming-harrington11':
            res[f'logrank_lifelines_{wt}'] = logrank_lifeline_survival_table(dfg, weightings='fleming-harrington', p=1, q=1).test_statistic
        elif wt == 'fleming-harrington01':
            res[f'logrank_lifelines_{wt}'] = logrank_lifeline_survival_table(dfg, weightings='fleming-harrington', p=0, q=1).test_statistic
        elif wt == 'fleming-harrington55':
            res[f'logrank_lifelines_{wt}'] = logrank_lifeline_survival_table(dfg, weightings='fleming-harrington', p=0.5, q=0.5).test_statistic
        else:
            res[f'logrank_lifelines_{wt}'] = logrank_lifeline_survival_table(dfg, weightings=wt).test_statistic
    return res


def simulate_null(T, N1, N2, lam0, nMonte, alternative='greater'):
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
    for _ in tqdm(range(nMonte)):
        Nt1, Nt2 = sample_survival_poisson(T, N1, N2, lam0, 0, 0)
        Ot1 = -np.diff(Nt1)
        Ot2 = -np.diff(Nt2)
        res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2,
                                  stbl=STBL, alternative=alternative, no_censoring=True)
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
                Nt1, Nt2 = sample_survival_poisson(T, N1, N2, lam0, eps, r)
                Ot1 = -np.diff(Nt1)
                Ot2 = -np.diff(Nt2)
                res1 = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2, stbl=STBL)
                res = pd.DataFrame(res1, index=[0])
                res['mu'] = mu
                res['eps'] = eps
                res['beta'] = beta
                res['itr'] = itr
                res['lam0'] = lam0
                df1 = df1.append(res, ignore_index=True)
    return df1


def evaluate_rare_and_weak(itr, T, N1, N2, lam0, beta, r):
    """
    order of arguments is important!
    evalaute an atomic experiment
    """

    eps = T ** (-beta)
    lam = lam0 * np.ones(T) / T

    Nt1, Nt2 = sample_survival_poisson(T, N1, N2, lam, eps, r)
    Ot1 = -np.diff(Nt1)
    Ot2 = -np.diff(Nt2)
    res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2,
                              randomized=False, alternative='both', stbl=STBL, no_censoring=True)
    return res


def main():
    T = 1000
    N1 = 5000
    N2 = 5000
    beta = .7

    print('Under null parameters')

    r = 0
    lam0 = 3

    res = evaluate_rare_and_weak(1, T, N1, N2, lam0, beta, r)
    print(res)

    print('Under alt. parameters')

    r = 2
    lam0 = 3

    res = evaluate_rare_and_weak(1, T, N1, N2, lam0, beta, r)
    print(res)


if __name__ == '__main__':
    main()
