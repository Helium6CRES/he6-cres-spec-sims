import numpy as np
import pandas as pd

import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.spec_calc.power_calc as pc

from .physics import *
from he6_cres_spec_sims.constants import *

class EventBuilder:
    """  Constructs a list of betas which are trapped within the detector volume
         (Doesn't hit waveguide walls && pitch angle is magnetically trapped)
    """
    def __init__(self, config):

        self.config = config
        self.physics = Physics(config)

    def run(self):

        print("~~~~~~~~~~~~EventBuilder Block~~~~~~~~~~~~~~\n")
        print("Constructing a set of betas:")

        # beta_num denotes the total number of betas produced in the trap.
        # event_num denotes the total number of trapped betas produced in the trap.
        beta_num = 0
        event_num = 0

        betas_to_simulate = self.config.physics.betas_to_simulate

        if betas_to_simulate < 0:
            raise ValueError("betas_to_simulate cannot be negative.")

        print( f"Simulating: num_betas:{betas_to_simulate}")

        #overwritten by DataFrame unless no trapped events
        events_df = None

        for beta_num in range(betas_to_simulate):
            if beta_num % 2500 == 0:
                print( f"\nBetas: {beta_num}/{betas_to_simulate - 1} simulated betas.")

            initial_position, initial_direction  = self.physics.generate_beta_position_direction()
            energy = self.physics.generate_beta_energy()

            single_beta_df = self.construct_untrapped_beta_df(initial_position, initial_direction, energy, beta_num, event_num)

            for event_index, event in single_beta_df.iterrows():
                single_event_df = self.fill_in_start_properties(event)

            #single_event_df (from fill_in_start_properties) should be a Series (single beta) so the following returns T/F
            if not single_event_df["trapped"].iloc[0]:
                continue

            if event_num == 0:
                events_df = single_event_df
            else:
                #note that event_num only gets put into events_df if event is trapped
                events_df = pd.concat([events_df, single_event_df], ignore_index=True)

            event_num += 1

        return events_df

    def construct_untrapped_beta_df( self, beta_position, beta_direction, beta_energy, beta_num, event_num):
        """ Computes e.g. guiding center position, range of cyclotron radii from beta parameters
        """
        # Initial beta position and direction.
        initial_rho_pos = beta_position[0]
        initial_phi_pos = beta_position[1]
        initial_zpos = beta_position[2]

        initial_theta = beta_direction[0]
        initial_phi_dir = beta_direction[1]

        initial_field = self.config.trap_profile.Bz(initial_rho_pos, initial_zpos)
        magnetic_moment = sc.magnetic_moment(beta_energy, initial_theta, initial_field)
        initial_radius = sc.cyc_radius(magnetic_moment, initial_field)

        # Given initial position, velocity vectors, compute guiding center position (x,y)
        # Note initial velocity vector (in x-y plane) is orthogonal to vector connecting guiding center to beta
        # \vec{r}_{GC} = \vec{r}_{init} - Rc \vec{n}_\perp, where \vec{v}_{init} \cdot \vec{n}_\perp = 0 with both unit length
        # Slightly inaccurate using Rc at beta position, and not at the guiding center (root-finding problem)
        center_x = initial_rho_pos * np.cos( initial_phi_pos / RAD_TO_DEG) - initial_radius * np.sin( initial_phi_dir / RAD_TO_DEG)
        center_y = initial_rho_pos * np.sin( initial_phi_pos / RAD_TO_DEG) + initial_radius * np.cos( initial_phi_dir / RAD_TO_DEG)

        rho_center = np.sqrt(center_x**2 + center_y**2)

        hamiltonian = sc.hamiltonian(beta_energy, rho_center, initial_zpos, self.config.trap_profile.voltage)

        initial_gamma = sc.gamma(beta_energy)

        initial_cos_theta = np.cos(initial_theta / RAD_TO_DEG)

        #all in eV
        initial_momentum = np.sqrt( (beta_energy + ME)**2 - ME**2)
        initial_pz = initial_momentum * initial_cos_theta

        track_properties = {
            # Conserved Quantities
            "hamiltonian": hamiltonian, #H = E + V (where E = γmc**2. H = E if no electric potential)
            "magnetic_moment": magnetic_moment,
            # Initial Kinematic Properties
            "start_energy": beta_energy, #note this is kinetic energy
            "start_momentum": initial_momentum,
            "start_gamma": initial_gamma,
            "start_rho": initial_rho_pos,
            "start_phi": initial_phi_pos,
            "start_z": initial_zpos,
            "start_theta": initial_theta,
            "start_cos_theta": initial_cos_theta,
            "start_phi_dir": initial_phi_dir,
            "start_field": initial_field,
            "start_radius": initial_radius,
            "start_pz": initial_pz,
            "start_guiding_center_x": center_x,
            "start_guiding_center_y": center_y,
            "start_guiding_center_rho": rho_center,
            "min_theta": np.nan,
            #"cos_center_theta": np.cos(center_theta / RAD_TO_DEG),
            "max_radius": np.nan,
            #Computed Properties
            "trapped": True,
            "z_turn_left": 0.0, #turning points of beta with z_turn_left < z_turn_right
            "z_turn_right": 0.0,
            "axial_freq": 0.0,
            "grad_b_freq": 0.0,
            "track_power": 0.0,
            "slope": 0.0,
            "track_length": 0.0,
            "start_time": np.nan,
            "start_freq": 0.0, #depends on average <B(z) / γ(z)>
            "start_time_in_trap_acq": np.nan,
            "b_avg": 0.0,
            #Final Kinematic Properties
            "end_energy": 0.0,
            "end_freq": 0.0,
            "end_time": np.nan,
            "end_time_in_trap_acq": np.nan,
            #Event IDs
            "beta_num": beta_num, # trapped + untrapped e± ID
            "event_num": event_num, # trapped e± ID
            "track_num": 0, # scatter-free time segment for a given event
            "acq_num": np.nan, # "second" of data in simulated run
            "trap_acq_num": np.nan,
        }

        beta_df = pd.DataFrame(track_properties, index=[beta_num])

        return beta_df

    def fill_in_start_properties(self, incomplete_scattered_events_df):
        """ Assigns calculated properties (e.g. axial frequency, power, slope, etc.)
            to beta with given (E, theta, rho) in the magnetic field profile
        """

        df = incomplete_scattered_events_df.copy()
        trap_profile = self.config.trap_profile
        main_field = self.config.eventbuilder.main_field
        decay_cell_radius = self.config.eventbuilder.decay_cell_radius

        # Calculate all relevant track parameters. Order matters here.

        #Rough wall effect calculation XXX Fix me for off axis gradient
        if ((df["start_guiding_center_rho"] + df["start_radius"] )  >= decay_cell_radius ):
            df["trapped"] = False
            return df.to_frame().T

        #determine whether trapped or not
        zKapton =  self.config.eventbuilder.kapton_zs

        # compute axial motion to determine whether it hits the walls or not. If so, exit & report as untrapped
        sol = sc.axial_trajectory(df["hamiltonian"], df["magnetic_moment"], df["start_guiding_center_rho"], df["start_z"], df["start_pz"], trap_profile, zKapton)

        #convenient to have default as trapped == True until shown to be untrapped
        #as there are multiple ways to be untrapped (walls || windows)
        if sol == False:
            df["trapped"] = False
            return df.to_frame().T

        #returns None, None if hits walls OR if axial period too long (>1 μs)
        turning_positions, axial_period = sc.get_turning_points_axial_period(sol)
        if turning_positions is None:
            df["trapped"] = False
            return df.to_frame().T

        zTurningPoints = [min(turning_positions), max(turning_positions)]

        b_avg = sc.get_b_avg(sol, df["start_guiding_center_rho"], axial_period, trap_profile)
        #Note for Penning trapping, this computes the mean <B/γ>, denominator can vary
        freq_start = sc.get_avg_cyc_freq(sol, df["hamiltonian"], df["start_guiding_center_rho"], axial_period, trap_profile)

        df["axial_freq"] = 1. / axial_period
        df["max_radius"] = sc.max_radius(df["magnetic_moment"], df["start_guiding_center_rho"], zTurningPoints, trap_profile)
        df["min_theta"] = sc.min_theta(df["magnetic_moment"], df["hamiltonian"], df["start_guiding_center_rho"], zTurningPoints, trap_profile)
        #more exact wall effect, using maximum cylotron radius
        if ((df["start_guiding_center_rho"] + df["max_radius"] )  >= decay_cell_radius ):
            df["trapped"] = False
            return df.to_frame().T

        #freq_start = sc.energy_to_freq(df["start_energy"], b_avg)
        #grad_b_freq = sc.grad_b_freq( df["energy"], df["center_theta"], df["rho_center"], trap_profile, axial_freq)

        track_radiated_power_te11 = (
            pc.power_calc(
                df["start_guiding_center_x"],
                df["start_guiding_center_y"],
                freq_start,
                b_avg,
                decay_cell_radius,
            )
        )

        track_radiated_power_tot = sc.power_larmor(b_avg, freq_start)
        slope = sc.df_dt( df["start_energy"], b_avg, track_radiated_power_tot)

        df["b_avg"] = b_avg
        df["start_freq"] = freq_start
        #df["grad_b_freq"] = grad_b_freq

        df["z_turn_left"] = zTurningPoints[0]
        df["z_turn_right"] = zTurningPoints[1]

        df["slope"] = slope
        df["track_power"] = track_radiated_power_te11

        #return a single rowed DataFrame
        return df.to_frame().T
