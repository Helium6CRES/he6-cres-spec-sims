import numpy as np

from he6_cres_spec_sims.spec_tools.beta_source.beta_source import BetaSource
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc

from he6_cres_spec_sims.constants import *


class Physics:
    """Creates distributions of beta kinematic parameters (position, velocity, time)
        according to the decaying isotope
    """

    def __init__(self, config, initialize_source=True):

        self.config = config
        self.bs = BetaSource(config)

        # distribution of rho positions [m]
        self.rho_distribution = self.config.dist_interface.get_distribution(self.config.physics.rho)

        # distribution of z positions [m]
        self.z_distribution = self.config.dist_interface.get_distribution(self.config.physics.z)

    def generate_beta_energy(self, beta_num):

        # Make this neater, have this function return energy in eV

        return self.bs.energy_array[beta_num]

    def generate_beta_position_direction(self):

        """
        Generates a random beta in the trap with pitch angle between
        min_theta and max_theta , and initial position (rho,0,z) between
        min_rho and max_rho and min_z and max_z.
        There should be a way to vectorize this, need to manage/ connect format of outputs
        """

        min_theta = self.config.physics["min_theta"] / RAD_TO_DEG
        max_theta = self.config.physics["max_theta"] / RAD_TO_DEG

        rho_initial = self.rho_distribution.generate()

        # No user choice (for now)
        phi_initial = 2 * PI * self.config.dist_interface.rng.uniform(0, 1) * RAD_TO_DEG

        z_initial = self.z_distribution.generate()

        u_min = (1 - np.cos(min_theta)) / 2
        u_max = (1 - np.cos(max_theta)) / 2

        sphere_theta_initial = np.arccos(1 - 2 * (self.config.dist_interface.rng.uniform(u_min, u_max))) * RAD_TO_DEG
        sphere_phi_initial = 2 * PI * self.config.dist_interface.rng.uniform(0, 1) * RAD_TO_DEG

        position = [rho_initial, phi_initial, z_initial]
        direction = [sphere_theta_initial, sphere_phi_initial]

        return position, direction

    def number_of_events(self):
        # determine number of events needed to simulate
        # TODO: option to do this using empirical beta rate to cres rate function,
        beta_rate = self.config.physics.beta_rate
        # bs_norm = self.bs.beta_spectrum.dNdE_norm
        # cres_ratio = integrate.quad(lambda x: self.bs.beta_spectrum.dNdE(x),
        #                             sc.freq_to_energy(self.config.physics.freq_acceptance_low,
        #                                                 self.config.eventbuilder.main_field),
        #                             sc.freq_to_energy(self.config.physics.freq_acceptance_high,
        #                                                 self.config.eventbuilder.main_field),
        #                             epsrel=1e-6
        #                             )[0]

        cres_ratio = self.bs.fraction_of_spectrum
        cres_rate = beta_rate*cres_ratio
        return cres_rate*self.config.daq.n_files*self.config.daq.spec_length
