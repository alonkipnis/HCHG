# Merge results in several files

from tqdm import tqdm
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='merge resukts')
    parser.add_argument('-i', type=str, help='input files', default='')
    parser.add_argument('-o', type=str, help='input files', default='results/merged.csv')
    args = parser.parse_args()
    #

    file_list = args.i
    df = pd.DataFrame()
    for fn in file_list.split():
        print(f"reading {fn}...")
        dfr = pd.read_csv(fn)
        print(f"Found {len(dfr)} records")
        df = pd.concat([df, dfr])
    print(f"Saving {len(df)} records to {args.o}...")
    df.to_csv(args.o)
    print('Done.')

if __name__ == '__main__':
    main()