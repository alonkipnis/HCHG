# %%
# !pip3 install two-sample-binomial
# !pip3 install multiple-hypothesis-testing
# !pip3 install lifelines
# !pip3 install multiHGtest

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize'] =  [8, 6]
mpl.style.use('ggplot')

import logging
from tqdm import tqdm

from SurvSet.data import SurvLoader


CRITVALS_FILENAME = "survset_critvals.json"

MIN_MAX_EVENTS = 15

MAX_CATEGORIES = 5 # don't split if 
FRAC_LOW = 0.1  # don't split if less than FRAC_LOW fraction of the samples are in one category
FRAC_HIGH = 1-FRAC_LOW

SIG_LEVEL = 0.05
GET_CRIEVAL_CRITVALS = False
nMonte = 10


logging.info("Finding relevant datasets...")
loader = SurvLoader()
lo_datasets = []


lo_ds_names = loader.df_ds.ds # List of available datasets and meta-info
acc = {}
logging.info(f"Datasets with at least {MIN_MAX_EVENTS} events in at least one time interval:")
for ds_name in lo_ds_names:
    df, ref = loader.load_dataset(ds_name=ds_name).values()
    if 'time2' in df.columns:
        df['diff'] = df['time2'] - df['time']
    else:
        df['diff'] = df['time']
    acc[ds_name] = df.groupby(['diff'])['event'].sum().max()
    if acc[ds_name] >= MIN_MAX_EVENTS:
        logging.info(f"{ds_name}")
        lo_datasets.append(ds_name)
        #df.groupby(['diff']).count()['event'].hist(bins=np.arange(30))
        #plt.title(f"{ds_name}")
        #plt.show()


# %% [markdown]
# ## Illustrate Curves

# %%
from test_gene_expression import test_gene

def sample_balanced_assignmet(T):
        """ Perfectly balanced assignment """
        a = np.random.rand(T)
        return a < np.median(a)
        
def simulate_null(data, nMonte, stbl=True, T=0, randomize=False):
        res = []
        logging.debug("Simulating null using real data...")
        df_test = data[['time', 'event']]
        for itr in tqdm(range(nMonte)):
                logging.debug(f'Sampling a random assignment')
                a = sample_balanced_assignmet(len(data))
                df_test = pd.DataFrame({'random_sample': a + 0,
                                        'time': df['time'],
                                        'event': df['event']})
                df_test['random_sample'] = df_test['random_sample'].astype(int)
                res_df = pd.DataFrame()

                r = test_gene(df_test, 'random_sample', T, stbl=stbl, randomize=randomize)
                res_df = pd.concat([res_df, r], axis=0)
                res_df['itr'] = itr
                res.append(res_df)
        return res

def simulate_critvals(data, nMonte):
        df0 = pd.concat(simulate_null(data, nMonte))
        return df0.quantile(1 - SIG_LEVEL)


import json


logging.info("Searching for critical values in ", CRITVALS_FILENAME)

try :
    with open(CRITVALS_FILENAME, 'r') as fp:
        critvals = json.load(fp)
except:
    logging.info("Critical values not found, simulating...")
    assert EVAL_CRITVALS == True, "Need to allow simulation of critical values"
    critvals = {}
    for ds_name in lo_datasets:
            print(f"Simulating for dataset {ds_name}...")
            df, ref = loader.load_dataset(ds_name=ds_name).values()
            critvals[ds_name] = simulate_critvals(df, nMonte).to_dict()
    logging.info("Saving critical values to ", CRITVALS_FILENAME)
    with open(CRITVALS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(critvals, f, ensure_ascii=False, indent=4)


lo_stats = ['hc_greater', 'log_rank_greater', 'log_rank_pval_greater',
            'hc_greater_rev', 'log_rank_greater_rev', 'log_rank_pval_greater_rev',
            'fisher_greater', 'min_p_greater', 'fisher_greater_rev', 'min_p_greater_rev',
            'logrank_lifelines_None',
            'logrank_lifelines_wilcoxon', 'logrank_lifelines_tarone-ware',
            'logrank_lifelines_peto',
            'logrank_lifelines_fleming-harrington55', 'logrank_lifelines_fleming-harrington11', 'logrank_lifelines_fleming-harrington01'
            ]

res = []
for ds_name in tqdm(lo_datasets):
    df, ref = loader.load_dataset(ds_name=ds_name).values()
    lo_facs = df.filter(like='fac_').columns
    for fac in lo_facs:
        # feature to partition
        lo_values = df[fac].unique()
        no_values = len(lo_values)
        if (no_values > MAX_CATEGORIES) or (no_values == 1): #
            continue
        if no_values == 2:
            lo_values = lo_values[:1]
        for value in lo_values:
            df['group'] = (df[fac] == value) + 0
            if (df['group'].mean() < FRAC_LOW) or (df['group'].mean() > FRAC_HIGH):
                continue
            r = test_gene(df, 'group', T=0, stbl=True, randomize=False)
            r['name'] = f"{ds_name}:{fac}:{value}"
            for stat_name in lo_stats:
                r[f"critval_{stat_name}"] = critvals[ds_name][stat_name]
            # r['critval_hc'] = critvals[ds_name]['hc_greater']
            # r['critval_lr'] = critvals[ds_name]['log_rank_greater']
            res.append(r)
df_res = pd.concat(res, axis=0)

# %%
# findi which instances exceeded the critical value
logging.info("For each instance, find all test statistics exceeding their critical values")
discoveries = {}
for stat_name in lo_stats:
    discoveries[stat_name] = (df_res[stat_name] > df_res['critval_'+stat_name])

# %%
logging.info("Report on the number of discoveries in every group")

for group1 in [['hc_greater', 'hc_greater_rev'], ['logrank_lifelines_None'], ['logrank_lifelines_wilcoxon'], ['logrank_lifelines_wilcoxon'],
               ['logrank_lifelines_tarone-ware'], ['logrank_lifelines_peto']]:
    print(f"\t{group1}")    
    to_exclude = ['fisher_greater', 'min_p_greater', 'fisher_greater_rev', 'min_p_greater_rev',
                'log_rank_pval_greater', 'log_rank_pval_greater_rev',
                'logrank_lifelines_fleming-harrington55',
                'logrank_lifelines_fleming-harrington11',
                'logrank_lifelines_fleming-harrington01',
                'log_rank_greater', 'log_rank_greater_rev',
                ]

    group2 = [s for s in lo_stats if s not in (group1 + to_exclude)]
    print(f"\t{group2}")
    #group2 = ['logrank_lifelines_None', 'logrank_lifelines_wilcoxon', 'logrank_lifelines_tarone-ware', 'logrank_lifelines_peto', 
            #'logrank_lifelines_fleming-harrington55', 'logrank_lifelines_fleming-harrington11', 'logrank_lifelines_fleming-harrington01'
    #]
    group_neutral = [s for s in lo_stats if s not in (group1 + group2 + to_exclude)]

    discoveries1 = [False] * len(df_res)
    discoveries2 = [False] * len(df_res)

    for sn in group1:
        discoveries1 = discoveries1 | discoveries[sn]
        
    for sn in group2:
        discoveries2 = discoveries2 | discoveries[sn]

    # for sn in group_neutral:
    #     discoveriesN = discoveriesN | discoveries[sn]


    print("Group1: ", group1)
    print("Group2: ", group2)
    print("Group N: ", group_neutral)
    print("Neither: ", to_exclude)

    print("Number of discoveries in Group1: ", np.sum(discoveries1))
    print("Number of discoveries in Group2: ", np.sum(discoveries2))
    print("Number of joint 1&2 discoveries: ", np.sum(discoveries1 & discoveries2))
    print("Number of unique discoveries 1: ", np.sum(discoveries1 & ~discoveries2))
    print("Number of unique discoveries 2: ", np.sum(discoveries2 & ~discoveries1))

    #print("Number of unique discoveries 1: ", np.sum(discoveries1 & ~discoveries2 & ~discoveriesN))
    #print("Number of unique discoveries 2: ", np.sum(discoveries2 & ~discoveries1 & ~discoveriesN))
    
    print("Discovered by Group1 but not Group2: (<dataset>:<factor>:<value>)")
    print(df_res['name'][discoveries1 & ~discoveries2].values)

