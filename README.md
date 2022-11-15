

# he6_cres_spec_sims

A package for simulating cres data.

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


## Done List: 

* Get field 