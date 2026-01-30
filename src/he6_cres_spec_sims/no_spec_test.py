import numpy as np
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, fixed
# import seaborn as sns
import sys

# Additional settings. 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Path to local imports. Alter to match your machine. 
sys.path.insert(1,"/home/lm11887/He6CRES/he6-cres-spec-sims/src")#/he6_cres_spec_sims")
print(sys.path)

# Local imports.
import he6_cres_spec_sims.experiment as exp
#fields = np.linspace(0.75, 3.25, 11)
fields = np.array([1.0])
#traps = np.around((fields/3),8)
traps = np.around(fields*1.8/3.25,6)
rand_seeds = np.array(fields*1213, dtype = int)
#rand_seeds = [None, None]
base_config_path = "/home/lm11887/He6CRES/spec_sims_results/local_experiments/local_base_config_example.yaml"

experiment_params = {
    "experiment_name": "ne_051424",
    "base_config_path": base_config_path,
    "events_to_simulate": 100,
    "betas_to_simulate": 100,
    "isotope": "Ne19",
    "rand_seeds": rand_seeds,
    "fields_T" : fields.tolist(), 
    "traps_A": traps.tolist()
}

for key, val in experiment_params.items(): 
    print("{}: {}".format(key, val))

ne19_exp = exp.Experiment(experiment_params)

