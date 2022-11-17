

# he6_cres_spec_sims

A package for simulating cres experiments over a variety of magnetic field values.


## Instructions: 

* **Get Dependencies**: 
	* *Instructions:* 
		* Log on to rocks. 
		* `cd /data/eliza4/he6_cres/simulation/`
		* `pip3 install -r he6-cres-spec-sims/requirements.txt` 
	* *Notes:*
		* May need to upgrade pip for the above to work on rocks. 
			* For Winston and I this worked: `pip3 install --upgrade pip`	
		* The following should contain all necessary python packages but if that isn't the case please let me (Drew) know. 
		* Be sure to add the `module load python-3.7.3` to your enviornment setup file or .bash_profile file so that you have access to python3.
		* The above must be done by each user, as it's the current users python packages that the scripts below will be utilizing. 

* **Simulate an Experiment**: 
	* *Instructions:* 
		* Log on to rocks. 
		* `cd /data/eliza4/he6_cres/simulation/`
		* Set up: 
			* Before running an experiment one needs a `.json` experiment config and a `.yaml` base config to both be in the following directory on rocks: `/data/eliza4/he6_cres/simulation/sim_results/experiments/`. See (TODO WHERE TO DOCUMENT THIS) somewhere for more details on what these two config files must contain. 
			* Here is how I copy those over from the examples shown in the repo (`he6-cres-spec-sims/config_files`). You should be able to do the same with minimal adjustment of paths.
				* `!scp /home/drew/He6CRES/he6-cres-spec-sims/config_files/* drewbyron@172.25.100.1:/data/eliza4/he6_cres/simulation/sim_results/experiments`
		* Initial run: 
			* `./he6-cres-spec-sims/he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/rocks_exp_config_example.json"`
		* Clean-up: 
			* `./he6-cres-spec-sims/he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/rocks_exp_config_example.json" -clean True`
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



## Notes as I build out the ability to run these simulations on rocks: 


* Get set up: 
	* Log on to rocks. 
	* `cd /data/eliza4/he6_cres/simulation/he6-cres-spec-sims`
	* Note: May need to upgrade pip. 
		* For Winston and I this worked: `pip3 install --upgrade pip`
		* For Heather the above didn't work and she needed to do the following: 
	* `pip3 install -r requirements.txt`

* Step 0: Getting an experiment running on rocks using qrsh: 
	* THIS WORKS. ON TO STEP 1
	* `qrsh`
	* `cd /data/eliza4/he6_cres/simulation`
	* `./he6-cres-spec-sims/he6_cres_spec_sims/run_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/exp_demo_nov2022.txt"`
	* Notes: 	
		* There is an example config file in the `he6-cres-spec-sims` repo under `config_files`. Copy that over under the experiments dir because the results will be written in the same directory that the config file you point to is. 

* Step 1: 
	* Getting each field sent to a different node. 
	* THIS WORKS ON TO STEP 2
	* `cd /data/eliza4/he6_cres/simulation`
	* `./he6-cres-spec-sims/he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/exp_demo_nov2022.txt"`

* Step 2: 
	* Simplest way to further parrelelize the simulations. Just call the simulation config again and again? How would that work?

	* 11/10/22: 
		* Working through this. Trying to make a simulation call an executable so that I can submit it as a job. This means the permissions have to be right. Make something an executable with this: 
			* `sudo chmod +x filename.py`
		* Trying to get the rocks experiment working here locally before porting it onto rocks. If this works I should be somewhat close. Maybe. 
		* Ok this seems to be working now. Nice. Next step is to get the qsub job submission working on rocks. Then need to think about how to clean up things if chunking up even further.  
			* This command works locally: 
				* `./he6-cres-spec-sims/he6_cres_spec_sims/run_rocks_experiment.py -exp "/media/drew/T7 Shield/spec_sims_results/experiments/test_experiment_11082022.txt"`
		* Trying the following command on rocks: 
			* `./he6-cres-spec-sims/he6_cres_spec_sims/run_rocks_experiment.py -exp "/data/eliza4/he6_cres/simulation/sim_results/experiments/exp_demo_nov2022.txt"`

	* It's working on a single node. 
	* Work on an run_experiment_rocks.py (rename other one) file to send out a bunch of different jobs to different nodes. One field per node.
	* Make logging output sensible with timestamps.
	* Test the limit of this method. How fast and how many betas will work?
	* do we need a clean-up?
* Step 2: 
	* How to chunk this up? Just run a few different experiments with different names ("\_0") and then write a clean-up script to combine them all into one. 


## Random Useful: 
* How to copy over my base config from local machine to rocks: 
	* `sudo scp /media/drew/T7 Shield/spec_sims_results/experiments/base_config_10202022.yaml drewbyron@172.25.100.1:/data/eliza4/he6_cres/simulation/sim_results/experiments/`
	* This isn't working. Not sure why. 


## To Do List: 

* Get the output to be a bit cleaner. Right now it gives 5 different csvs but really the dmtracks contains all the info doesn't it?
* Want to have clear instructions for running an experiment locally and running an experiment on rocks. And for how to analyze the results. 
* The name of the experiment doesn't match the name of the .txt which is a little annoying. Should make that the way it works. (on rocks)
* The output to the log will be too damn much rn. Need an option to run in debug mode or something?

* 11/15/22: 
	* Error when running 1e6 betas. 
		* return self.bs.energy_array[beta_num]IndexError: index 1000000 is out of bounds for axis 0 with size 1000000.
	* Maybe just rerun everything as is then put a sim_index col when combining everything? That may be easier to impliment and keep track of in the immediate. 
	* Make the .txt into a json? That may be easier for people to understand the format of in terms of it being a dictionary? 
	* Should I add a date in the logs somewhere? Or even in the name of the experiment?
	* Get rid of sim_ in the rocks dictionaries as this is redundant and makes navigation harder. 
	* Maybe a directory should be made for all the logs associated with a given experiment, this would be better I think. 

	* AFTER BREAKFAST: 
		* First get cleanup working to first order.
		* Why are print statements in the logs not matching my new code? May need to hard reset the remote?
		* Get things working to first order by the end of the next block. Keep with the 45 minute chunks. 
	* How to clean up??
		* Ok how to actually do this? I need to consider this carefully. Should I make the clean-up more complex and make it match what the data class wants or make the data class different to accomadate this? 
			* Hmm. It almost seems easier to do in the data class BUT what if it was 20 different directories. It's clunky not to combine those. 
		* Ok also right now each one of these is an exact copy of the rest as I'm not changing the seeds at all... That's an issue. 

	* Current approach. Commit to it then see how it works: 
		* I make the base experiment plus some number of copies. The base has no subscript. 
		* Then I go through and add the contents of the copies dmtracks to the original one by globbing through the different directories. 
			* Be sure to add a `exp_copy` col or something to the dmtracks df. 
		* Then copy to my local comp and make sure the experiment class still works. Edit as needed. 
		* Stay with it, you got this. 
	* **For tomorrow:** 
		* Work on getting the exp dict copied to a local machine (reuse the rocks analysis code), and make sure the output makes sense and everything. Then need to go on to cleaning and commenting everthing. 
		* Get this project to the DONE point.
	* **For today (11/15/22)**: 
		* Have tested the local and remote versions of the simulation run and the results visualizations by this evening. 
	* **After breakfast**: 
		* Go through an entire rocks experiment and document here as you go -> Making the instructions as you go. 
		* Then go through an entire local experiment and do the same. 
		* Then try to get some nice visualizations in the readme. 
		* Then work on cleaning things up and docstrings. 
		* Head up. This is worth doing and worth doing well. You're doing great. But need to stay with it. 
	* **START HERE! After lunch**: 
		* Change naming conventions so they make more sense. 
		* Get rid of the other stuff besides tracks. It's simpler like this. Make the experiment dir behave this way as well.
		* See if things break when you crank up the number of simulations/ number of betas simulated. 
		* Put demo .ipynb in the repo. Make one for a local experiment and a rocks experiment. 
		* Keep going with the readme for the local experiment 


## Done List: 

* Get field 