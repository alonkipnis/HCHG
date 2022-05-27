# to do:
# either transform into a model GenerateConfiguration
# or only load confirutations from a .csv file
import numpy as np
import itertools
import yaml
import pandas as pd
from typing import Dict
import logging


def read_params(params) -> Dict:
    """
    Args:
        params: either a dictionary or a path to a .csv file

    Return:
        dictionary of parameters

    """
    if type(params) == str :
            param_file = params
            logging.info(f" Reading parameters from {param_file}.")
            with open(param_file) as file:
                out = yaml.load(file, Loader=yaml.FullLoader)
    elif type(params) == dict:
        out = params
    else:
        raise ValueError(" could not understand :params: format")
        out = None
    return out


def load_configurations(params) -> pd.DataFrame:
    """
    Args:
        params: either a dictionary or a path to a .csv file

    Return:
        dataframe with each row indicating a different experiment configuration

    """

    if type(params) == str:  # params is a filename, either CSV of YML
            param_file = params

            if '.csv' in param_file:  # case CSV
                logging.info("Reading configurations from CSV file...")
                df_conf = pf.read_csv(param_file)

            if 'yaml' in param_file or 'yml' in param_file: # case YML
                logging.info("Reading configurations from YAML file...")
                with open(param_file) as file:
                    param_dict = yaml.load(file, Loader=yaml.FullLoader)

                logging.info(f"Generating configurations based on {param_dict}...")
                df_conf = pd.DataFrame(generate_conf_from_dict(param_dict))

    elif type(params) == dict:  # params is a dictionary
        logging.info("Generating configurations based on {params}...")
        df_conf = pd.DataFrame(generate_conf_from_dict(params))
    else:
        raise ValueError(" could not understand :params: format")
        param_dict = None

    return df_conf


def generate_conf_from_dict(conf_dict):
    
    def gen_series(var) :
        rec = conf_dict[var]
        if rec['type'] == 'range(float)':
            tp = conf_dict[var].get('float')
            return np.linspace(rec['min'], rec['max'], int(rec['length'])).astype(tp)
        if rec['type'] == 'range(int)':
            return np.arange(rec['min'], rec['max']+1).astype(int)
        if rec['type'] == 'list(int)':
            return [int(li) for li in rec['values']]
        if rec['type'] == 'list(float)':
            return [float(li) for li in rec['values']]

    srs = []
    for var_name in conf_dict['names']:
        srs += [gen_series(var_name)]

    for s in list(itertools.product(*srs)):
        yield dict(zip(conf_dict['names'], s))


def main():
    param_file = 'params.yaml'
    with open(param_file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    conf = pd.DataFrame(generate_conf_from_dict(params))
    print(f"Generated {len(conf)} configurations.")
    print(conf)

if __name__ == '__main__':
    main()
