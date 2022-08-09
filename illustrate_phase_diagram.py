import numpy as np
import pandas as pd
import scipy
import seaborn
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO)


import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['figure.figsize'] = [8, 6]
mpl.style.use('ggplot')


def q95(x):
    """
    Compute the 95-th percentile of the vector x
    """
    if x.dtypes.kind == 'O':
        return np.nan
    else:
        return pd.Series.quantile(x, .95)


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

            # we check rate when both sides succeeded. These are not good outcomes
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
            fn = out_filename + '_' + tsn + str(c[0]) + ".png"
            plt.savefig(fn)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Illustrate Results')
    parser.add_argument('-i', type=str, help='results file', default='results.csv')
    parser.add_argument('-n', type=str, help='null data', default="")
    parser.add_argument('-o', type=str, help='output dir', default='Figs/')
    args = parser.parse_args()
    #
    logging.info(f"Reading from {args.i}...")
    df = pd.read_csv(args.i).filter(regex='^((?!Unnamed).)*$')
    df1 = df[df.r > 0]

    logging.info(f"Found {len(args.i)} records.")

    if args.n == "":
        results_null_file = args.i
    else:
        results_null_file = args.n

    df0 = pd.read_csv(results_null_file).filter(regex='^((?!Unnamed).)*$')
    df0 = df0[df0.r == 0]

    illustrate_phase_diagrams(df1, df0, out_filename=args.o+'phase_diagram')

if __name__ == '__main__':
    main()




