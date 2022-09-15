# Analyze data obtained after sampling from the null distributon
# report critical values and estimated null parameters

import numpy as np
import scipy
from tqdm import tqdm
import pandas as pd
import argparse
import re
import scipy.stats


def quantile(x, q):
    """
    Compute the q-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, q)

def q95(x):
    return quantile(x, .95)


def std_95(x):
    """
        Standard error in estimating the 95 quantile based on the vector of
        measurements x using the Maritz-Jarrett method.
    """
    return scipy.stats.mstats.mjci(x, prob = [0.95])[0]


def main():
    parser = argparse.ArgumentParser(description='Analyze SCANB')
    parser.add_argument('-i', type=str, help='input file', default='')
    args = parser.parse_args()
    #

    df0 = pd.read_csv(args.i)

    T = int(re.findall(r'T([0-9]+)', args.i)[0])

    precision = 5

    dsp = df0.agg([q95, 'mean', 'std']).filter(
        ['log_rank_greater', 'hc_greater', 'x0', 'y0', 'lam1', 'lam2'])
    dsp.loc['std_95'] = [std_95(df0[c]) for c in dsp]

    print(f"Printing statistics from {args.i} with T={T}...")
    print("================================================")

    print(np.round(dsp, precision))

    if 'lam1' in dsp:
        m = (dsp.loc['mean', 'lam1'] + dsp.loc['mean', 'lam2']) / 2
        s = np.sqrt((dsp.loc['std', 'lam1'] ** 2 + dsp.loc['std', 'lam2'] ** 2) / 2)
        print("lam * T = ", np.round(m * T, precision))
        print("SE(lam*T) = ", np.round(s * T, precision))


if __name__ == '__main__':
    main()