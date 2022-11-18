
<p align="center"><img width="30%" src="/demo/readme_imgs/he6-cres_logo.png" /></p>

--------------------------------------------------------------------------------
# he6_cres_spec_sims

A package for simulating cres experiments over a variety of magnetic field values.

--------------------------------------------------------------------------------
### Simulate an experiment then make interactive plots of cres track features!

<p align="center"><img width="40%" src="/demo/readme_imgs/plot_stuff.png" />              <img width="50%" src="/demo/readme_imgs/make_plot.png" /></p>

<p align="center"><img width="40%" src="/demo/readme_imgs/plot_stuff_1.png" />              <img width="50%" src="/demo/readme_imgs/make_plot_1.png" /></p>

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
			* Before running an experiment one needs a `.json` experiment config and a `.yaml` base config to both be in the following directory on rocks: `/data/eliza4/he6_cres/simulation/sim_results/experiments/`. (TODO: Document the config files and the output fields.)
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
		* Clean up:
			* In the clean-up phase the different copies of the experiment that are produced by the run are combined into one directory that can then be copied onto a local machine for analysis. 
			* In the example used above where we have 5 copies of an experiment spanning 3 fields each with 1000 betas simulated, all of the resultant `.csvs` containing track info for the 5 copies is combined into one directory. 
		* General: 
			* The logs will get very clogged up with different `.txt` files. Occassionally just delete the contents of `simulation/sim_logs`.
	* **Analyzing simulation results**: 
		* *Intructions:* 
			* Use the class `SimulationResults` from the `results.py` module (need to have a local copy of the repo) to grab the experiment results. 
			* Example code to be run in ipynb or within a script: 
				* `sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")` (enables you to run this anywhere, change paths to be your own)
				* `from he6_cres_spec_sims.results import SimulationResults`
				* `local_dir = "/media/drew/T7 Shield/spec_sims_results/rocks_experiments"`
				* `sim_exp_name = "rocks_exp_config_example"`
				* `sim = SimulationResults(local_dir = local_dir, sim_exp_name = sim_exp_name )`
			* Explore all that the `sim` instance contains: `sim.results.__dict__`
			* All simulated tracks (as pd.DataFrame) are here: `sim.results.tracks`
		* *Notes:*
			* See the demo notebook for a full illustration of the above: `he6-cres-spec-sims/demo/rocks_sim_experiment_demo.ipynb`.

--------------------------------------------------------------------------------

## Instructions for running simulations locally: 

* **Get dependencies**: 
	* *Instructions:* 
		* Navigate into desired parent directory.
		* Clone the repo, or `git pull` if you already have it. A hard reset to the remote may be necessary if you have an old version. You should be on the `develop` branch (default).  
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
			* `./run_local_experiment.py -exp "/media/drew/T7 Shield/spec_sims_results/local_experiments/local_exp_config_example.json"`

	* *Notes:*
		* Run experiment:
			* No clean-up is necessary. 
			* The only difference between the config files used locally and on rocks is that there is no `experiment_copies` field in the local `.json` config. The base config (`.yaml`) is identical. 

	* **Analyzing simulation results**: 
		* *Intructions:* 
			* Use the class `ExperimentResults` from the `experiment.py` module to grab the experiment results. 
			* Example code to be run in ipynb or within a script: 
				* `sys.path.append("/home/drew/He6CRES/he6-cres-spec-sims/")` (enables you to run this anywhere, change paths to be your own)
				* `from he6_cres_spec_sims.experiment import ExpResults`
				* `experiment_config_path = "/media/drew/T7 Shield/spec_sims_results/local_experiments/local_exp_config_example/local_exp_config_example_exp.yaml"`
				* `sim = ExpResults.load(experiment_config_path =experiment_config_path  )`
			* Explore all that the `sim` instance contains: `sim.__dict__`
			* All simulated tracks (as pd.DataFrame) are here: `sim.tracks`
		* *Notes:*
			* See the demo notebook for a full illustration of the above: `he6-cres-spec-sims/demo/local_sim_experiment_demo.ipynb`.

--------------------------------------------------------------------------------

## Documentation for simulation config files: 

--------------------------------------------------------------------------------

### Experiment config (.json)

One can find an example of this config file here: `/he6-cres-spec-sims/config_files/rocks_exp_config_example.json`

#### An example of it's contents: 

{"experiment_copies": 3, "experiment_name": "defaults to .json name", "base_config_path": "/data/eliza4/he6_cres/simulation/sim_results/experiments/rocks_base_config_example.yaml", "isotope": "He6", "events_to_simulate": -1, "betas_to_simulate": 1e2, "rand_seeds": [4062, 3759, 3456, 3153, 2850, 2547, 2244, 1941, 1638, 1335, 1032], "fields_T": [3.25, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75], "traps_A": [1.8, 1.661538, 1.523077, 1.384615, 1.246154, 1.107692, 0.969231, 0.830769, 0.692308, 0.553846, 0.4153845]}

#### Explanation of the fields it must contain:

experiment_copies: 
experiment_name: 
base_config_path: 
isotope: 
events_to_simulate: 
betas_to_simulate: 
rand_seeds: 
fields_T:
traps_A: 

#### To build your own given a query from he6-cres database: 

* `import json`
* `json.dump(my_exp_dict, 'path_to_exp_dict.json')`

--------------------------------------------------------------------------------

### Base config (.yaml)

One can find an example of this config file here: `/he6-cres-spec-sims/config_files/rocks_base_config_example.yaml`

#### Explanation of the fields it must contain:

Settings: 
    rand_seed: 234

Physics:
    events_to_simulate : "inf"
    betas_to_simulate : 100
    energy_spectrum: 
        beta_source: "Ne19"
    freq_acceptance_high: 18.0e+9
    freq_acceptance_low: 19.1e+9
    min_rho : 0.0
    max_rho : 5.78e-3
    min_z : -2.5e-3
    max_z : 2.5e-3
    min_theta : 89.0
    max_theta : 90.0

EventBuilder:
    main_field : 1.700468
    trap_current : .1
    decay_cell_radius : 5.78e-3

SegmentBuilder: 
    mean_track_length : 10.0e-3
    jump_size_eV : 14
    jump_std_eV : 5
    pitch_angle_costheta_std : 0.0
    jump_num_max : 0

BandBuilder: 
    sideband_num: 1
    frac_total_segment_power_cut : 0.01
    harmonic_sidebands: True
    magnetic_modulation: False 

TrackBuilder:
    run_length: 60.0e-6

DMTrackBuilder:
    mixer_freq: 17.9e+9

Daq:
    daq_freqbw: 1.2e+9
    freq_bins: 32768
    fft_per_slice: 2
    band_power_override: 1.0e-16
    gain_override: 1.0
    
SpecBuilder:
    specfile_name: "example_spec_file"

--------------------------------------------------------------------------------
## Documentation for tracks output: 

List of all columns output by simulations: 

['energy', 'gamma', 'energy_stop', 'initial_rho_pos', 'initial_phi_pos',
       'initial_zpos', 'initial_theta', 'cos_initial_theta', 'initial_phi_dir',
       'center_theta', 'cos_center_theta', 'initial_field', 'initial_radius',
       'center_x', 'center_y', 'rho_center', 'trapped_initial_theta',
       'max_radius', 'min_radius', 'avg_cycl_freq', 'b_avg', 'freq_stop',
       'zmax', 'axial_freq', 'mod_index', 'segment_power', 'slope',
       'segment_length', 'band_power', 'band_num', 'segment_num', 'event_num',
       'beta_num', 'fraction_of_spectrum', 'energy_accept_high',
       'energy_accept_low', 'gamma_accept_high', 'gamma_accept_low',
       'time_start', 'time_stop', 'freq_start', 'exp_copy', 'simulation_num',
       'field', 'trap_current'] 

TODO: Fill in the above. 

--------------------------------------------------------------------------------

## Important notes: 

* In case of emergency please break glass. No actually just email me at wbyron@uw.edu. 
* The `develop` branch is tested and working on rocks and locally as of 11/16/22.


## To Dos (11/16/22): 

* Work on making a visual readme. Get some demos of the functionality. 
* Note somewhere that  
* Clean up the code and make docstrings! You got this. 
* Test a rocks run with a lot of stats to see what breaks.
* Work on documenting the modules (quickly) with docstrings. 
* Then move on to making sure that katydid on rocks still works without the pip install. For now I am uninstalling the package with a pip uninstall. This will break katydid on rocks!! So I need to go fix that once I'm done with this


## Done List: 
* Merge this branch into develop. 


## Imports!

* Ok so generally I'm doing things right but things get wierd when you try to run a script inside your package. That generally causes issues. So try to get run_rocks_exp also working outside of the package. 