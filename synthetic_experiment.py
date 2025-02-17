# %%
# !pip3 install two-sample-binomial
# !pip3 install multiple-hypothesis-testing

import numpy as np
from tqdm import tqdm
import pandas as pd
from survival import evaluate_test_stats

import sys
sys.path.append('phase_transition_experiment/')
from sample_survival_poisson import sample_survival_poisson

seed = 0
np.random.seed(seed)

# %% [markdown]
# ## Synthetic experiment
# 
# Sample data from rare and weak exponential decay and compare several test statistics

# %%
import hashlib

def hash_parameters(parameters, digits=10):
    # Convert the parameters to a string representation
    parameters_str = str(parameters)

    # Hash the string representation using SHA-256
    hash_object = hashlib.sha256(parameters_str.encode())

    # Get the hexadecimal representation of the hash value
    hash_value = hash_object.hexdigest()

    return hash_value[:digits]

# 
T = 84
N1 = N2 = 1000
lam0 = 1.5
alternative = 'two-sided'
lam = lam0 * np.ones(T) / T

# ##### Null data
# 
# We use empirical 0.95 (for two-sided) and 0.975 (for one-sided) quantiles of the test statistics obtained over ``nMonte`` samples from the null model. 
# %%
# ##### Null data
nMonte_null = 10000
params_null = dict(alternative=alternative, T = T, N1 = N1, N2 = N2, lam0 = lam0, nMonte = nMonte_null)
print("Null Parameters: ", params_null)
fn = f"results/synthetic_data_null_{hash_parameters(params_null)}.csv"

# try reading file. If it exists, then load it. Else, simulate and save it.
try:
    print(f"Reading null data from file {fn}")
    df0 = pd.read_csv(fn, index_col=0)
    results0 = df0.to_dict()
    print("Columns of df0:", df0.columns)
    print(f"Found results of {len(df0)} Monte Carlo iterations with {len(results0)} satistics.")
except FileNotFoundError:
    print(f"File {fn} not found. Simulating null data and saving to file.")
    results0 = {}
    for itr in tqdm(range(nMonte_null)):
        Nt1, Nt2 = sample_survival_poisson(T, N1, N2, lam, 0, 0)
        Ot1 = -np.diff(Nt1).astype(int)
        Ot2 = -np.diff(Nt2).astype(int)
        res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2, alternative=alternative)
        results0[itr] = res
    pd.DataFrame(results0).T.to_csv(fn)
    print(f"Saved null data to file {fn}")
df0 = pd.read_csv(fn, index_col=0)


beta = .7
r = 1.2
nMonte = 1000
alternative = 'two-sided'
params=dict(alternative=alternative, beta=beta,r=r, T = T, N1 = N1, N2 = N2, lam0 = lam0, nMonte = nMonte)
print("Parameters: ", params)
eps = T ** (-beta)


##### Alternative data
fn = f"results/synthetic_data_T{T}_N{N1}_{hash_parameters(params)}.csv"
try:
    print(f"Trying to read alternative data from file {fn}")
    df1 = pd.read_csv(fn, index_col=0)
    results = df1.to_dict()
    print(f"Found results of {len(df1)} Monte Carlo iterations with {len(results)} satistics.")
except FileNotFoundError:
    print(f"File {fn} not found. Simulating alternative data and saving to file.")
    results = {}
    for itr in tqdm(range(nMonte)):
        Nt1, Nt2 = sample_survival_poisson(T, N1, N2, lam, eps, r)
        Ot1 = -np.diff(Nt1).astype(int)
        Ot2 = -np.diff(Nt2).astype(int)
        res = evaluate_test_stats(Nt1[:-1], Nt2[:-1], Ot1, Ot2,
                                randomized=False, alternative=alternative)
        results[itr] = res
    pd.DataFrame(results).T.to_csv(fn)
    print(f"Saved alternative data to file {fn}")

df1 = pd.read_csv(fn, index_col=0)


# %%
lo_stats = df1.columns
# you can specify critical values for each test if you do not wish to use null simulations
crit_vals_table = {'chisq_test_stat': 2.6288}


print("Columns of df0:", df0.columns)
print("Stats in lo_stats:", lo_stats)

print("Forming two-sided statistic...")
stat_1sided = ['hc_greater', 'min_p_greater', 'log_rank_greater', 'fisher_greater']
#lo_stats.append('hc_2sided') 

print("Evalauting number of discoveries...")
alpha = 0.05
crit_vals = {}
no_dicoveries = {}
rate_dicoveries = {}
for st in lo_stats:
    if '_rev' in st:
        continue
    if st in crit_vals_table:
        crit_vals[st] = crit_vals_table[st]
    else:
        crit_vals[st] = df0[st].quantile(1 - alpha)
    if st in stat_1sided: # for one-sided test, must not report on cases of an effect in both ways
        no_dicoveries[st] = np.sum((df1[st] > crit_vals[st]))
    else:
        no_dicoveries[st] = np.sum(df1[st] > crit_vals[st])
    rate_dicoveries[st] = no_dicoveries[st] / len(df1)


print("Rate of discoveries:")
print(rate_dicoveries)
print("Latex table:")
df_res = pd.DataFrame.from_dict(rate_dicoveries, orient='index').sort_values([0],ascending=False)
print(df_res.to_latex())
df_res.T.to_csv("results/synthetic_experiment_rate_discoveries.csv")




