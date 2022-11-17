
<p align="center"><img width="15%" src="/demo/readme_imgs/he6-cres_logo.png" /></p>

--------------------------------------------------------------------------------
# he6_cres_spec_sims

A package for simulating cres experiments over a variety of magnetic field values.

--------------------------------------------------------------------------------
### Make nice interactive plots of cres track features

<p align="center"><img width="40%" src="/demo/readme_imgs/choose_features.png" />              <img width="40%" src="/demo/readme_imgs/make_plot.png" /></p>

--------------------------------------------------------------------------------
## Instructions for running simulations on CENPA cluster (rocks): 

* **Get dependencies**: 
	* *Instructions:* 
		* Log on to rocks. 
		* `cd /data/eliza4/he6_cres/simulation/`
		* `pip3 install -r he6-cres-spec-sims/requirements.txt` 
	* *Notes:*
		* May need to upgrade pip for the above to work on rocks. 
			* For Winston and I this worked: `pip3 install --upgrade pip`	
		* The following should contain all necessary python packages but if that isn't the case please let Drew Byron know. 
		* Be sure to add the `module load python-3.7.3` to your enviornment setup file or .bash_profile file so that you have access to python3.
		* The above must be done by each user, as it's the current users python packages that the scripts below will be utilizing. 

* **Simulate an experiment**: 
	* *Instructions:* 
		* Log on to rocks. 
		* `cd /data/eliza4/he6_cres/simulation/he6-cres-spec-sims`
		* Set up: 
			* Before running an experiment one needs a `.json` experiment config and a `.yaml` base config to both be in the following directory on rocks: `/data/eliza4/he6_cres/simulation/sim_results/experiments/`. See (TODO WHERE TO DOCUMENT THIS) somewhere for more details on what these two config files must contain. 
			* Here is how I copy those over from the examples shown in the repo (`he6-cres-spec-sims/config_files`). You should be able to do the same with minimal adjustment of paths.
				* `!scp /home/drew/He6CRES/he6-cres-spec-sims/config_files/rocks* drewbyron@172.25.100.1:/data/eliza4/he6_cres/simulation/sim_results/experiments`
		* Initial run: 
			* `./he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/rocks_exp_config_example.json"`
		* Clean-up: 
			* `./he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/rocks_exp_config_example.json" -clean True` 
	* *Notes:*
		* Initial run:
			* `-exp` (str): Specify the path to the json file that contains the specific attributes (in the form of a python dictionary) of the simulated experiment. See the docstring for the `run_rocks_experiment.py` module for a complete description of all attributes that the `.json` must contain.
			* Say one made a `.json` experiment config locally based on some queries to the he6 postgreSQL database. You could copy that to the rocks `sim_results/experiments` directory with a command like this: 
				* `!scp /media/drew/T7\ Shield/spec_sims_results/rocks_experiments/exp_demo_nov2022.json drewbyron@172.25.100.1:/data/eliza4/he6_cres/simulation/sim_results/experiments`
			* Here `experiment_copies` number of independent (unique random seeds) but otherwise identical experiments are run over rocks. The experiment attribute `experiment_copies` is specified in the `.json` config file. It is parallelized such that each field specified in each copy is sent to a different node. So for example if the `.json` config had these attributes: `{"experiment_copies": 5, "beta_num": 1000, "fields_T": [1.0, 2.0, 3.0]}`, then 5 copies x 3 fields = 15 nodes would each simulate 1000 betas. 
			* There needs to be a base config file in the `/sim_results/experiments` directory that the `.json` config file points to. One can copy a local `.yaml` file over from your local machine with a command like this: 
				* TODO: FILL THIS IN. ALSO put all the base components necessary along with a demo in the repo somewhere.  
		* Clean up:
			* In the clean-up phase the different copies of the experiment that are produced by the run are combined into one directory that can then be copied onto a local machine for analysis. 
			* In the example used above where we have 5 copies of an experiment spanning 3 fields each with 1000 betas simulated, all of the resultant `.csvs` containing track info for the 5 copies is combined into one directory. 
	* **Analyzing simulation results**: 
		* *Intructions:* 
			* Use the class `SimulationResults` from the `results.py` module (need to have a local copy of the repo) to grab the experiment results. 
			* Example code to be run in ipynb or within a script: 
				* `sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")`
				* `from he6_cres_spec_sims.results import SimulationResults`
				* `local_dir = "/media/drew/T7 Shield/spec_sims_results/rocks_experiments"`
				* `sim_exp_name = "rocks_exp_config_example"`
				* `sim = SimulationResults(local_dir = local_dir, sim_exp_name = sim_exp_name )`
			* Explore all that the `sim` instance contains: `sim.results.__dict__`
			* All simulated tracks (as pd.DataFrame) are here: `sim.results.tracks`
		* *Notes:*
			* See the demo notebook for a full illustration of the above: `he6-cres-spec-sims/demo/rocks_sim_experiment_demo.ipynb`.

## Instructions for running simulations locally: 

* **Get dependencies**: 
	* *Instructions:* 
		* Navigate into desired parent directory.
		* Clone the repo, or `git pull` if you already have it. A hard reset to the remote may be necessary if you have an old version. (NEED THE DEV BRNACH TO BE WORKING.)  
			* `git clone git@github.com:Helium6CRES/he6-cres-spec-sims.git`
		* `pip3 install -r he6-cres-spec-sims/requirements.txt` 
	* *Notes:*
		* You will need pip3 for the above to work. 

* **Simulate an experiment**: 
	* *Instructions:* 
		* Navigate into your `he6-cres-spec-sims` parent directory.
		* Set up: 
			* Before running an experiment one needs a `.json` experiment config and a `.yaml` base config to both be in a directory suited for simulation results. For me this is in an external drive. 
			* Here is how I copy those over from the examples shown in the repo (`he6-cres-spec-sims/config_files`). You should be able to do the same with minimal adjustment of paths.
				* `cp /home/drew/He6CRES/he6-cres-spec-sims/config_files/local* /media/drew/T7\ Shield/spec_sims_results/local_experiments`
			* The `base_config_path` field in the `.json` experiment config needs to be manually changed to point at the `.yaml` base config file. Change this path. 
		* Run experiment: 
			* `./he6-cres-spec-sims/run_local_experiment.py -exp "/media/drew/T7 Shield/spec_sims_results/local_experiments/local_exp_config_example.json"`

	* *Notes:*
		* Run experiment:
			* No clean-up is necessary. 
			* The only difference between the config files used locally and on rocks is that there is no `experiment_copies` field in the local `.json` config. The base config (`.yaml`) is identical. 

	* **Analyzing simulation results**: 
		* *Intructions:* 
			* Use the class `ExperimentResults` from the `experiment.py` module to grab the experiment results. 
			* Example code to be run in ipynb or within a script: 
				* `sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")`
				* `from he6_cres_spec_sims.experiment import ExpResults`
				* `experiment_config_path = "/media/drew/T7 Shield/spec_sims_results/local_experiments/local_exp_config_example/local_exp_config_example_exp.yaml"`
				* `sim = ExpResults.load(experiment_config_path =experiment_config_path  )`
			* Explore all that the `sim` instance contains: `sim.__dict__`
			* All simulated tracks (as pd.DataFrame) are here: `sim.tracks`
		* *Notes:*
			* See the demo notebook for a full illustration of the above: `he6-cres-spec-sims/demo/local_sim_experiment_demo.ipynb`.

## To Dos (11/17/22): 

* Work on making a visual readme. Get some demos of the functionality. 
* Note somewhere that the logs should be deleted every once in a while. 
* Clean up the code and make docstrings! You got this. 
* Merge this branch into develop. 
* Test a rocks run with a lot of stats to see what breaks.
* Then move on to making sure that katydid on rocks still works without the pip install. For now I am uninstalling the package with a pip uninstall. This will break katydid on rocks!! So I need to go fix that once I'm done with this


## Done List: 
* A lot. 


## Imports!

* Ok so generally I'm doing things right but things get wierd when you try to run a script inside your package. That generally causes issues. So try to get run_rocks_exp also working outside of the package. 