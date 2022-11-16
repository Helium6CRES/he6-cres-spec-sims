#!/usr/bin/env python3
from itertools import compress
from pathlib import Path
import shutil
import subprocess
import pathlib
import paramiko
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing

import he6_cres_spec_sims.experiment as exp

class SimulationResults:

    """
    Notes:
    * Ok this should use the experiment class that's already written?
    """

    def __init__(
        self,
        local_dir,
        sim_exp_name,
        include_sampled_gammas=False,
        rebuild_sim_exp_dir=False,
        rocks_username="drewbyron",
        rocks_IP="172.25.100.1",
    ):

        # Attributes.
        self.local_dir = Path(local_dir)
        self.sim_exp_name = sim_exp_name
        self.include_sampled_gammas = include_sampled_gammas
        self.rebuild_sim_exp_dir = rebuild_sim_exp_dir
        self.rocks_username = rocks_username
        self.rocks_IP = rocks_IP

        self.rocks_base_path = Path(
            "/data/eliza4/he6_cres/simulation/sim_results/experiments"
        )

        self.sim_exp_dir_rocks = self.rocks_base_path / Path(self.sim_exp_name)
        self.sim_exp_dir_loc = self.local_dir / Path(self.sim_exp_name)
        self.sim_exp_json_rocks = self.rocks_base_path / Path(
            self.sim_exp_name
        ).with_suffix(".json")
        self.sim_exp_json_loc = self.local_dir / Path(self.sim_exp_name).with_suffix(
            ".json"
        )

        # Setting this as a default until we fill it with the dataframes and such.
        self.results = None

        # Step 0. Copy sim_exp results over to a local directory.
        self.build_local_sim_exp()

        # Step 1. Load the results using the Experiment class.
        self.load_experiment()

    def build_local_sim_exp(self):

        if self.sim_exp_dir_loc.exists():

            if self.rebuild_sim_exp_dir:
                print("Rebuilding local simulation experiment dir.")
                shutil.rmtree(str(self.sim_exp_dir_loc))
                self.copy_remote_sim_exp_dir()
                self.copy_remote_sim_exp_json()

            else:
                print("Keeping existing experiment directory.")
        else:
            self.copy_remote_sim_exp_dir()
            self.copy_remote_sim_exp_json()

        return None

    def copy_remote_sim_exp_dir(self):
        print(
            f"\nCopying sim results for experiment {self.sim_exp_name} from rocks.\
             This may take a few minutes.\n"
        )
        scp_run_list = [
            "scp",
            "-r",
            f"{self.rocks_username}@{self.rocks_IP}:{str(self.sim_exp_dir_rocks)}",
            str(self.local_dir),
        ]
        self.execute(scp_run_list)

        return None

    def copy_remote_sim_exp_json(self):
        print(
            f"\nCopying json for experiment {self.sim_exp_name} from rocks.\
            This may take a few minutes.\n"
        )
        scp_run_list = [
            "scp",
            f"{self.rocks_username}@{self.rocks_IP}:{str(self.sim_exp_json_rocks)}",
            str(self.sim_exp_json_loc),
        ]
        self.execute(scp_run_list)

        return None

    def execute(self, cmd):

        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for line in popen.stdout:
            print(line, end="")
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, cmd)
        return None

    def load_experiment(self):
        self.exp_config_path = self.sim_exp_dir_loc / Path(
            self.sim_exp_name + "_exp.yaml"
        )

        if not self.exp_config_path.exists():
            raise UserWarning(
                f"The experiment config file doesn't exist: {self.exp_config_path}"
            )
        exp_config_path = "/media/drew/T7 Shield/spec_sims_results/rocks_experiments/exp_demo_nov2022/exp_demo_nov2022_exp.yaml"
        self.results = exp.ExpResults.load(
            experiment_config_path=self.exp_config_path,
            include_sampled_gammas=self.include_sampled_gammas,
        )

        return None

    def scatter(
        self,
        column_1,
        column_2,
        fix_field=False,
        field_value=0,
        scatt_settings={
            "figsize": (12, 4),
            "colors": ["b", "r", "g", "c", "m", "k"],
            "hist_bins": 200,
            "markersize": 0.4,
            "alpha": 1.0,
        },
    ):

        if self.results == None:
            raise ValueError(
                f"Experiment not loaded. No results to display scatter plots for."
            )

        df = self.results.experiment_results

        if fix_field:
            condition = df.field == field_value
            df = df[condition]

        plt.close("all")
        fig0, ax0 = plt.subplots(figsize=scatt_settings["figsize"])

        ax0.set_title("Scatter: {} vs {}".format(column_1, column_2))
        ax0.set_xlabel("{}".format(column_1))
        ax0.set_ylabel("{}".format(column_2))

        # Scatter Plots
        ax0.plot(
            df[column_1],
            df[column_2],
            "o",
            markersize=scatt_settings["markersize"],
            alpha=scatt_settings["alpha"],
            color=scatt_settings["colors"][0],
        )

        plt.show()

        fig1, ax1 = plt.subplots(figsize=scatt_settings["figsize"])

        ax1.set_title("Histogram. x_col: {}".format(column_1))
        ax1.set_xlabel("{}".format(column_1))

        # Histogram.
        ax1.hist(
            df[column_1],
            bins=scatt_settings["hist_bins"],
            color=scatt_settings["colors"][1],
        )

        plt.show()

        fig2, ax2 = plt.subplots(figsize=scatt_settings["figsize"])

        ax2.set_title("Histogram. y_col: {}".format(column_2))
        ax2.set_xlabel("{}".format(column_2))

        # Histogram.
        ax2.hist(
            df[column_2],
            bins=scatt_settings["hist_bins"],
            color=scatt_settings["colors"][1],
        )

        plt.show()

        return None
