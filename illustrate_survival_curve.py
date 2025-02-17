# !pip3 install two-sample-binomial
# !pip3 install multiple-hypothesis-testing
from multitest import MultiTest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from phase_transition_experiment.sample_survival_poisson import *
from survival import evaluate_test_stats, multi_pvals


def find_changes(Nt1, Nt2, Ot1, Ot2, stbl=False, gamma=.35):
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    mt = MultiTest(pvals[pvals<=1], stbl=stbl) 
    hc, hct = mt.hc(gamma=gamma)
    return pvals <= hct

def illustrate_survival_curve(Nt1, Nt2, Ot1, Ot2, Ct1, Ct2, stbl=True,
                               show_HCT=True, randomize_HC=False):
    
    stats = evaluate_test_stats(Nt1, Nt2, Ot1, Ot2, stbl=stbl, randomize=randomize_HC)
        
    pvals = multi_pvals(Nt1, Nt2, Ot1, Ot2, randomize=False)
    pvals_rev = multi_pvals(Nt2, Nt1, Ot2, Ot1, randomize=False)
    mt = MultiTest(pvals[pvals<1], stbl=stbl) 
    hc, _ = mt.hc(gamma=.2)
    
    fpval = find_changes(Nt1, Nt2, Ot1, Ot2, stbl=stbl)
    
    dfg = pd.DataFrame()
    dfg['at_risk:0'] = Nt1
    dfg['at_risk:1'] = Nt2
    dfg['events1'] = Ot1
    dfg['events2'] = Ot2
    dfg['pvalue'] = pvals
    dfg['pvalue_rev'] = pvals_rev
    dfg['Survival Proportion X'] = (Nt1 - Ct1) / (np.max(Nt1) - Ct1)
    dfg['Survival Proportion Y'] = (Nt2 - Ct2) / (np.max(Nt2) - Ct2)
    
    df_disp = dfg[fpval]
    
    plt.step(dfg.index, dfg['Survival Proportion X'], 'b', where='pre')
    plt.step(dfg.index, dfg['Survival Proportion Y'], 'r', where='pre')
    s1 = 100 * Ct1 / np.max(Ct1) # size of censorship markers
    s2 = 100 * Ct2 / np.max(Ct2)
    plt.scatter(dfg.index[Ct1 > 0], dfg.loc[Ct1 > 0,'Survival Proportion X'],
                marker='+', c='k',
                s=s1[Ct1 > 0], alpha=.6)
    plt.scatter(dfg.index[Ct2 > 0], dfg.loc[Ct2 > 0,'Survival Proportion Y'],
                marker='+', c='k',
                s=s2[Ct2 > 0], alpha=.6)
    
    plt.legend([r'$\hat{S}_x$', r'$\hat{S}_y$'], fontsize=16, loc=1)
    
    if show_HCT:
        plt.bar(dfg.index[:len(fpval)], fpval, color='k', alpha=.2, width=.5)
    hc = stats['hc_greater']
    #hc_rev = stats_rev['hc_greater']
    logrank = stats['log_rank_greater']

    st_HC = r"$\mathrm{HC}$"
    #st_HC_rev = r"$\mathrm{Rev}(\mathrm{HC})$"
    st_LR = r"$\mathrm{LR}$"
    #plt.title(rf"{st_HC}={np.round(hc,2)}, {st_HC_rev}={np.round(hc_rev,2)}, {st_LR}={np.round(logrank,2)}")
    plt.title(rf"{st_HC}={np.round(hc,2)}, {st_LR}={np.round(logrank,2)}")
    #plt.title(f"{gene_name}, (HC={np.round(stats['hc_greater'],2)}, Log-rank={np.round(stats['log_rank_greater'],2)})")
    plt.ylabel('proportion', fontsize=16)
    plt.xlabel(r'$t$ [time]', fontsize=16)
    
    return df_disp, dfg