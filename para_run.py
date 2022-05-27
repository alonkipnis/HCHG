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
    

class ParaRun :
    def __init__(self, eval_func, params='params.yaml') :
        
        self._out = pd.DataFrame()
        self._func = eval_func
        
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
        job_ids = []
        itr = 0
        for params in tqdm(self._configurations.itertuples(index=False)):
            r = self._func(*params)
            results.append(r)
            itr += 1
            job_ids += [itr]
        
        self._out = self._configurations.join(pd.DataFrame(results), how='left')
        logging.info(f" Completed.")

    def Dask_run(self, client) :
        """
        Apply atomic expriment function to each row in configuration table

        Args:
        -----
        func    atomic experiment function

        """

        logging.info(f" Running on Dask.")

        logging.info(" Mapping to futures...")

        variables = self._configurations.columns.tolist()
        self._configurations.loc[:,'job_id'] = 'null'
        futures=[]

        df_vars = self._configurations.filter(items=variables)
        for r in df_vars.iterrows() :
            fut = client.submit(self._func, **r[1])
            self._configurations.loc[r[0], 'job_id'] = fut.key
            futures += [fut]
        logging.info(" Gathering...")
        
        progress(futures)
        
        keys = [fut.key for fut in futures]
        results = pd.DataFrame(client.gather(futures), index=keys)
        logging.info(" Terminating client...")
        client.close()
    
        self._out = self._configurations.set_index('job_id').join(results, how='left')
        

    def to_file(self, filename="results.csv") :
        if self._out.empty :
            logging.warning(" No output." 
            "Did the experiment complete running?")
        if self._configurations.empty :
            logging.warning(" No configuration detected."
            "Did call gen_conf_table() ")

        logging.info(f" Saving results...")
        logging.info(f" Saved {len(self._out)} records in {filename}.")
        self._out.to_csv(filename)    


def main() :
    parser = argparse.ArgumentParser(description='Launch experiment')
    parser.add_argument('-o', type=str, help='output file', default='')
    parser.add_argument('-p', type=str, help='yaml parameters file.', default='params.yaml')
    parser.add_argument('--dask', action='store_true')
    parser.add_argument('--address', type=str, default="")
    args = parser.parse_args()
    #
    

    if args.dask :
        logging.info(f" Using Dask:")
        if args.address == "" :
            logging.info(f" Starting a local cluster")
            client = Client()
        else :
            logging.info(f" Connecting to existing cluster at {args.address}")
            client = Client(args.address)
        logging.info(f" Dashboard at {client.dashboard_link}")
        exper = ParaRun(evaluate, args.p)
        exper.Dask_run(client)
            
    else :
        exper = ParaRun(evaluate, args.p)
        exper.run()

    output_filename=args.o
    if output_filename == "":
        import pdb; pdb.set_trace()
        dig = hash(str(exper._params))
        output_filename = f"results_{dig}.csv"

    exper.to_file(output_filename)
    

if __name__ == '__main__':
    main()
