import numpy as np
import itertools
import yaml
import pandas as pd


def generate(params):
    return gen_normal(params)

def gen_normal(variables) :
    """
    Parameters for 1-sample normal means experiment

    """
    def gen_series(var) :

        rec = variables[var]
        if rec['type'] == 'range':
            tp = variables[var].get('float')
            return np.linspace(rec['min'], rec['max'],
                         int(rec['length'])).astype(tp)
        if rec['type'] == 'int_range':
            return np.arange(rec['min'], rec['max']+1).astype(int)
        if rec['type'] == 'list':
            return rec['values']


    srs = []
    for var_name in variables['names']:
        srs += [gen_series(var_name)]

    for s in list(itertools.product(*srs)):
        yield dict(zip(variables['names'], s))


def main():
    param_file = 'params.yaml'
    with open(param_file) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    conf = pd.DataFrame(generate(params['variables']))
    print(f"Generated {len(conf)} configurations.")
    print(conf)

if __name__ == '__main__':
    main()
