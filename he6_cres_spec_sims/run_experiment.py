#!/usr/bin/env python3
import json
import sys
import argparse

# Local imports.
# from . import experiment as exp
# Path to local imports.
sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")

import he6_cres_spec_sims.experiment as exp



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

    print(f"\n\n\n Beginning simulation.\n\n\n")

    sim_experiment_params = json.load(open(dict_path))
    for key, val in sim_experiment_params.items():
        print("{}: {}".format(key, val))

    sim_experiment = exp.Experiment(sim_experiment_params)

    print(f"\n\n\n Done running simulation. {sim_experiment_params}")

    return None


if __name__ == "__main__":
    main()
