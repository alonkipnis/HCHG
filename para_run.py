import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from typing import Dict

import logging

logging.basicConfig(level=logging.INFO)
import argparse

import dask
import time
from dask.distributed import Client, progress

from survival import evaluate
from configurations import load_configurations


class ParaRun(object):
    """
    ParaRun module for running emberisingly parallelizable computational expriments
    ParaRun receives an atomic experiment function and a table in which 
    each row represnets one configuration of the expeirments. The atomic 
    experiment function returns a dictionary representing the result of one 
    experiments. All varaibles in an experiment's configuration and result are 
    arranged as one row in ParaRun._out. 

    Args:
        atomic_experiment_function:   function comprising a single experiment
                                      it returns a dictionary
        params:    experiment configurations. Can be one of: 
        - 'CSV' file containing the table configurations
        - 'YML' file containing instructions for generating a table of configurations
        - Python dictionary containing instructions for generating a table of configurations
        
    Methods:
        run:   apply atomic expriment function to each row in configuration table and stores
            the results as a large table in which each row contains configuration varaibles
            and the result of the experiment involving this configuration
        Dask_run:   same as ParaRun.run, but use Dask.distributed scaling manager 
        to_file:   store the results to a CSV file


    Examples:
        import numpy as np

        atomic_experiment_function = lambda x,y : np.sqrt(x**2 + y**2)
        configuration_dict = {'names'}

        $ 


    """

    def __init__(self, atomic_experiment_function, params='params.yaml'):

        self._out = pd.DataFrame()
        self._func = atomic_experiment_function

        self._configurations = load_configurations(params)
        logging.info(f"Found {len(self._configurations)} configurations.")

    def run(self):
        """
        Apply atomic expriment function to each row in configuration table

        Args:
        -----
        func    atomic experiment function
        """
        logging.info(f" Running...")

        results = []
        itr = 0
        for params in tqdm(self._configurations.itertuples(index=False)):
            r = self._func(*params)
            results.append(r)
            itr += 1

        self._out = self._configurations.join(pd.DataFrame(results), how='left')
        logging.info(f" Completed.")

    def Dask_run(self, client):
        """
        Apply atomic expriment function to each row in configuration table

        Args:
        -----
        func    atomic experiment function

        """

        logging.info(f" Running on Dask.")

        logging.info(" Mapping to futures...")

        variables = self._configurations.columns.tolist()
        self._configurations.loc[:, 'job_id'] = 'null'
        futures = []

        df_vars = self._configurations.filter(items=variables)

        for r in df_vars.itertuples():  # iterate of dataframe's rows
            index, variables = r[0], r[1:]  # r is a tuple (index, var1, var2, ...)
            fut = client.submit(self._func, *variables)
            self._configurations.loc[index, 'job_id'] = fut.key
            futures += [fut]
        logging.info(" Gathering...")

        progress(futures)

        keys = [fut.key for fut in futures]
        results = pd.DataFrame(client.gather(futures), index=keys)
        logging.info(" Terminating client...")
        client.close()

        self._out = self._configurations.set_index('job_id').join(results, how='left')

    def to_file(self, filename="results.csv"):
        if self._out.empty:
            logging.warning(" No output."
                            "Did the experiment complete running?")
        if self._configurations.empty:
            logging.warning(" No configuration detected."
                            "Did call gen_conf_table() ")

        logging.info(f" Saving results...")
        logging.info(f" Saved {len(self._out)} records in {filename}.")
        self._out.to_csv(filename)


def main():
    parser = argparse.ArgumentParser(description='Launch experiment')
    parser.add_argument('-o', type=str, help='output file', default='')
    parser.add_argument('-p', type=str, help='yaml parameters file.', default='params.yaml')
    parser.add_argument('--dask', action='store_true')
    parser.add_argument('--address', type=str, default="")
    args = parser.parse_args()
    #

    if args.dask:
        logging.info(f" Using Dask:")
        if args.address == "":
            logging.info(f" Starting a local cluster")
            client = Client()
            print(client)
        else:
            logging.info(f" Connecting to existing cluster at {args.address}")
            client = Client(args.address)
        logging.info(f" Dashboard at {client.dashboard_link}")
        exper = ParaRun(evaluate, args.p)
        exper.Dask_run(client)

    else:
        exper = ParaRun(evaluate, args.p)
        exper.run()

    output_filename = args.o
    if output_filename == "":
        dig = hash(str(self._configurations))
        output_filename = f"results_{dig}.csv"

    exper.to_file(output_filename)


if __name__ == '__main__':
    main()
