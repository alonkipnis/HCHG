import numpy as np
import pandas as pd
import TwoSampleHC
import logging

from scipy.stats import poisson, norm, chisquare, binom, chi2_contingency
from scipy.spatial.distance import cosine

import sys
import os

from TwoSampleHC import (HC, binom_test_two_sided_random, two_sample_pvals, 
    binom_test_two_sided, poisson_test_random, binom_var_test_df, binom_var_test)


def two_sample_chi_square(c1, c2, lambda_="pearson"):
    """returns the Chi-Square score of the two samples c1 and c2
     (representing counts). Null cells are ignored. 

    Args: 
     c1, c2 : list of integers
        representing two 1-way contingency tables
     lambda_ : one of :
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [Rf6c2a1ea428c-3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   
    
    Returns
    -------
    chisq : score 
        score divided by degree of freedom. 
        this normalization is useful in comparing multiple
        chi-squared scores. See Ch. 9.6.2 in 
        Yvonne M. M. Bishop, Stephen E. Fienberg, and Paul 
        W. Holland ``Discrete Multivariate Analysis''  
    log_pval : log of p-value
    """
    
    if (sum(c1) == 0) or (sum(c2) == 0) :
        return np.nan, 1
    else :
        obs = np.array([c1, c2])
        if lambda_ in ['mod-log-likelihood',
                         'freeman-tukey',
                          'neyman'] :
            obs_nz = obs[:, (obs[0]!=0) & (obs[1]!=0)]
        else :
            obs_nz = obs[:, (obs[0]!=0) | (obs[1]!=0)]

        chisq, pval, dof, exp = chi2_contingency(
                                    obs_nz, lambda_=lambda_)
        if pval == 0:
            Lpval = -np.inf
        else :
            Lpval = np.log(pval)
        return chisq / dof, Lpval
        

def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)

def sample_from_mixture(lmd0, lmd1, eps) :
    N = len(lmd0)
    idcs = np.random.rand(N) < eps
    #idcs = np.random.choice(np.arange(N), k)
    lmd = np.array(lmd0.copy())
    lmd[idcs] = np.array(lmd1)[idcs]
    return np.random.poisson(lam=lmd)

def power_law(n, xi) :
    p = np.arange(1.,n+1) ** (-xi)
    return p / p.sum()


def two_sample_poisson(n, N, be, r, xi, metric = 'Hellinger') :
    logging.debug(f"Evaluating with: n={n}, N={N}, be={be}, r={r}, xi={xi}")
    P = power_law(N, xi)
    
    mu = r * np.log(N) / n / 2 
    ep = N ** -be
    if metric == 'Hellinger' :
      QP = (np.sqrt(P) + np.sqrt(mu))**2

    if metric == 'ChiSq' :
      QP = P + 2 * np.sqrt(P * mu)

    if metric == 'proportional' :
      QP = P *( 1 + r * np.log(N))

    if metric == 'power' :
      QP = P * (np.log(N) ** r)

    smp1 = sample_from_mixture(n*P, n*QP, ep)
    smp2 = sample_from_mixture(n*P, n*QP, ep)

    stbl = False
    gamma = 0.25

    def filter_pvals(pv) :
        return pv[~((smp1 == 0) & (smp2 == 0))]

    def test_stats(pv) :
        if len(pv) > 0 :
            hc, _ = HC(pv[pv < 1], stbl=stbl).HC(gamma=gamma)
            min_pv = pv.min()
        else :
            hc = np.nan
            min_pv = np.nan
        return hc, min_pv
    
    pv_rand = two_sample_pvals(smp1, smp2, randomize=True, sym=True)
    pv = two_sample_pvals(smp1, smp2, randomize=False)
    pv_one_NR = poisson_test_random(smp1, n*P)

    # two sample non-random
    hc_rand, MinPv_rand = test_stats(filter_pvals(pv_rand))

    hc, MinPv = test_stats(filter_pvals(pv))
    hc_one_NR, MinPv_one_NR = test_stats(filter_pvals(pv_one_NR))

    chisq, _ = two_sample_chi_square(smp1, smp2)

    cos = cosine_sim(smp1, smp2)

    #pv_stripes = binom_var_test(smp1, smp2, sym=True).values
    #hc_stripes, MinPv_stripes = test_stats(pv_stripes)

    return { 'HC_random' : hc_rand,
             'minPv_random' : MinPv_rand,
             'HC' : hc,
             'minPv' : MinPv,
             'HC_one_NR' : hc_one_NR,
             'minPv_one_NR' : MinPv_one_NR,
             'chisq' : chisq,
             'cos' : cos,
             #'hc_stripes' : hc_stripes,
             #'minPv_stripes' : MinPv_stripes,
             }


def two_sample_normal_pvals(n, be, r, sig):
    """
    2-sample normal means experiment
    """

    mu = np.sqrt(2 * r * np.log(n))
    ep = n ** -be

    Z1 = sample_from_normal_mix(n, ep/2, mu, sig)
    Z2 = sample_from_normal_mix(n, ep/2, mu, sig)

    Z = (Z1 - Z2)/np.sqrt(2)
    pvals = 2*norm.cdf(- np.abs(Z))
    return pvals

def sample_from_normal_mix(n, ep, mu, sig):
    """
    Sample from `n` times from a normal 
    mixture with sparsity parameter controlled
    by `be` and intensity of non-null mixture
    is determined by `r` with std `sig`
    """

    idcs = np.random.rand(int(n)) < ep 
    Z = np.random.randn(int(n))
    Z[idcs] = sig*Z[idcs] + mu
    return Z

def one_sample_normal_pvals(n, be, r, sig):
    """
    1-sample normal means experiment
    """
    mu = np.sqrt(2 * r * np.log(n))
    ep = n ** -be

    Z = sample_from_normal_mix(n, ep, mu, sig)
    return norm.sf(Z)


def test_fdr(pvals):
    n_features = len(pvals)
    sv = np.sort(pvals)
    uu = np.arange(1, n_features + 1) #/  n_features
    return np.min(sv / uu)


def evaluate(itr, n, beta, r, sig) :
    logging.debug(f"Evaluating with: n={n}, beta={beta}, r={r}, sig={sig}")

    gamma = .25

    pvals = one_sample_normal_pvals(n, beta, r, sig)
    
    # tests:
    fisher = -2*np.log(pvals).mean()
    _hc = HC(pvals) 
    hc,_ = _hc.HC()
    hcstar,_ = _hc.HCstar()
    minP = -2*np.log(pvals.min())
    bj = _hc.berk_jones()
    fdr = -2*np.log(test_fdr(pvals))

    return {
    'hc' : hc,
    'hcstar' : hcstar,
    'fisher' : fisher,
    'minP' : minP,
    'bj' : bj,
    'fdr' : fdr
    }
    


