#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path

# # Path to local imports.
# sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

# Local imports.
import rocks_experiment as exp

def main():
    """
    DOCUMENT

    """

    # Parse command line arguments.
    par = argparse.ArgumentParser()
    arg = par.add_argument
    arg(
        "-exp",
        "--exp_dict_path",
        type=str,
        help="path (str) to the pickled dict (.txt) that defines the run conditions of the experiment.",
    )

    args = par.parse_args()
    run_experiment(args.exp_dict_path)

    return None


def run_experiment(dict_path):

    print(f"\n\n\nBeginning simulation.\n\n\n")

    # Load the .txt file into a dictionary. 
    sim_experiment_params = json.load(open(dict_path))

    default_exp_copies = 1
    exp_copies = sim_experiment_params.pop('experiment_copies', 1)
    # Make the experiment name match the name of the .txt file. 
    for copy in range(exp_copies): 
        experiment_name = Path(dict_path).stem + f"_{copy}"
        sim_experiment_params["experiment_name"] = experiment_name

        print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(f"Experiment Copy: {copy}\n")
        print("Summary of simulation experiment:\n")
        for key, val in sim_experiment_params.items():
            print("{}: {}".format(key, val))

        sim_experiment = exp.RocksExperiment(sim_experiment_params)

        print(f"\n\n\n Submitted jobs for simulation experiment.")

    return None

def clean_up_experiment(dict_path): 

    # First check to see that all the copies are present. 
    # Then combine them all into one directory without any prefix. 
    # Then delete the old directories. 
    return None



if __name__ == "__main__":
    main()
