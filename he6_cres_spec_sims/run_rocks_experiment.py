#!/usr/bin/env python3
import json
import sys
import argparse
from pathlib import Path
from shutil import rmtree
from glob import glob
import pandas as pd


# Local imports.
import he6_cres_spec_sims.rocks_experiment as exp


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
    arg(
        "-clean",
        "--clean_up",
        type=bool,
        default=False,
        help="if true then the simulation experiment has already been run and a clean-up will occur. ",
    )

    args = par.parse_args()

    if args.clean_up:
        clean_up_experiment(args.exp_dict_path)

    else:
        run_experiment(args.exp_dict_path)

    return None


def run_experiment(dict_path):

    print(f"\n\n\nSimulation rocks experiment.\n\n\n")

    # Load the .json file into a dictionary.
    sim_experiment_params = json.load(open(dict_path))

    exp_copies_dirs = get_exp_dirs(dict_path)

    # Make the experiment name match the name of the .json file.
    for copy, exp_dir in enumerate(exp_copies_dirs):

        experiment_name = exp_dir.stem

        sim_experiment_params["experiment_name"] = experiment_name

        # Need to change the seeds of each copy or else they are identical.
        sim_experiment_params["rand_seeds"] = [
            seed + copy * 11 for seed in sim_experiment_params["rand_seeds"]
        ]

        print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(f"Experiment Copy: {copy}\n")
        print("Summary of simulation experiment:\n")
        for key, val in sim_experiment_params.items():
            print("{}: {}".format(key, val))

        sim_experiment = exp.RocksExperiment(sim_experiment_params)

        print(f"\n\n\nSubmitted jobs for simulation experiment.")

    return None


def clean_up_experiment(dict_path):

    print(f"\n\n\nClean up.\n\n\n")

    # Get the directory paths to the experiment (+ copies).
    exp_copies_dirs = get_exp_dirs(dict_path)

    # Check to see that all the exp copies are present.
    all_copies_exist = all([edir.exists() for edir in exp_copies_dirs])
    if not all_copies_exist:
        raise UserWarning(
            f"Not all of the {len(exp_copies_dirs)} copies are present: {exp_copies_dirs}"
        )

    # Merge track csvs into exp_dir.
    if len(exp_copies_dirs) > 1:
        merge_csvs(exp_copies_dirs)
        del_dirs(exp_copies_dirs[1:])
    else:
        print("No clean-up necessary, only one copy was created.")

    # Then delete the copy directories.

    return None


def build_exp_dir(exp_dir):

    if not exp_dir.exists():
        print(f"Making resultant directory: {exp_dir}")
        exp_dir.mkdir()

    else:
        print("Directory already exists: {} ".format(exp_dir))
        print(
            "\nCAREFUL: Continuing will delete the contents of the above directory.\n"
        )
        input("Press Enter to continue...")
        rmtree(exp_dir)
        exp_dir.mkdir()

    return None


def del_dirs(dirs):
    for exp_dir in dirs:
        print(f"Recursively deleting dir: {exp_dir}")
        rmtree(exp_dir)
    return None


def get_exp_dirs(dict_path):

    exp_dirs = []

    # Load the .json file into a dictionary.
    sim_experiment_params = json.load(open(dict_path))

    default_exp_copies = 1
    exp_copies = sim_experiment_params.pop("experiment_copies", 1)

    # Make the experiment name match the name of the .txt file.
    for copy in range(exp_copies):
        if copy == 0:
            experiment_name = Path(dict_path).stem
        else:
            experiment_name = Path(dict_path).stem + f"_{copy}"

        sim_experiment_params["experiment_name"] = experiment_name
        exp_dir = exp.get_experiment_dir(sim_experiment_params)
        exp_dirs.append(Path(exp_dir))

    return exp_dirs


def merge_csvs(exp_copies_dirs):

    # Make the output only tracks?? Yes, for now.

    # Step 0:  Gather all relevant paths.
    tracks_name_in_sim = "dmtracks"
    tracks_paths_lists = []
    for exp_dir in exp_copies_dirs:
        tracks_paths_lists.append(
            sorted(list(exp_dir.glob(f"*/{tracks_name_in_sim}.csv")))
        )

    # QUESTION: Will this clean-up break with large csvs? 
    for tracks_path_list in list(zip(*tracks_paths_lists)):
        print(len(tracks_path_list))
        tracks_dfs = [
            pd.read_csv(tracks_path, index_col=0) for tracks_path in tracks_path_list
        ]

        # Add in the copy index. 
        for copy in range(len(tracks_dfs)):
            tracks_dfs[copy]["exp_copy"] = copy

        tracks_df = pd.concat(tracks_dfs, ignore_index=True)

        # This should work as long as the sorted ordering remains sensible.
        # In that I am assuming the first list item has no _0, _1, and so on.
        # The default mode is to overwrite ('w'). Just adding it for clarity.
        tracks_df.to_csv(tracks_path_list[0], mode="w")

        lens = [len(df) for df in tracks_dfs]
        print("\nCombining set of tracks_dfs.\n")
        print("\nlengths: ", lens)
        print("\nsum: ", sum(lens))
        print("\nlen single file (sanity check): ", len(tracks_df))
        print("\ntracks index: ", tracks_df.index)
        print("\ntracks cols: ", tracks_df.columns)

    # print("this:/n", zip(*tracks_paths_lists))

    # resultant_tracks_path = exp_dir / Path(f"dmtracks.csv")
    # tracks_path_list = [edir / Path(f"dmtracks.csv") for edir in exp_copies_dirs]
    # tracks_exist = [path.is_file() for path in tracks_path_list]

    # print(tracks_path_list, tracks_exist)

    # if not all(tracks_exist):
    #     raise UserWarning(
    #         f"{sum(tracks_exist)}/{len(tracks_exist)} track csvs are present."
    #     )

    # tracks_dfs = [
    #     pd.read_csv(tracks_path, index_col=0) for tracks_path in tracks_path_list
    # ]
    # tracks_df = pd.concat(tracks_dfs, ignore_index=True)
    # lens = [len(df) for df in tracks_dfs]
    # print("\nCombining set of tracks_dfs.\n")
    # print("lengths: ", lens)
    # print("sum: ", sum(lens))
    # print("len single file (sanity check): ", len(tracks_df))
    # print("tracks index: ", tracks_df.index)
    # print("tracks cols: ", tracks_df.columns)

    # tracks_df.to_csv(resultant_tracks_path)
    # events_df.to_csv(self.events_df_path)

    return None


if __name__ == "__main__":
    main()
