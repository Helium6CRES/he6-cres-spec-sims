from he6_cres_spec_sims.spec_tools.beta_source.beta_source import BetaSource
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc


class Physics:
    """Creates distributions of beta kinematic parameters (position, velocity, time)
        according to the decaying isotope
    """

    def __init__(self, config, initialize_source=True):

        self.config = config
        self.bs = BetaSource(config)

    def generate_beta_energy(self, beta_num):

        # Make this neater, have this function return energy in eV

        return self.bs.energy_array[beta_num]

    def generate_beta_position_direction(self):

        # Could maybe improve this by not generating a new one each time,
        # it could be vectorized the way the energy is...

        position, direction = sc.random_beta_generator( self.config.physics, self.config.dist_interface.rng)

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
