import numpy as np
import pandas as pd

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
        beta_num = 0

        betas_to_simulate = self.config.physics.betas_to_simulate

        if betas_to_simulate < 0:
            raise ValueError("betas_to_simulate cannot be negative.")

        print( f"Simulating: num_betas:{betas_to_simulate}")

        for beta_num in range(betas_to_simulate):
            if beta_num % 2500 == 0:
                print( f"\nBetas: {beta_num}/{betas_to_simulate - 1} simulated betas.")

            initial_position, initial_direction  = self.physics.generate_beta_position_direction()
            energy = self.physics.generate_beta_energy()

            single_beta_df = self.construct_untrapped_beta_df(initial_position, initial_direction, energy, beta_num)

            if beta_num == 0:
                betas_df = single_beta_df

            else:
                betas_df = pd.concat([betas_df, single_beta_df], ignore_index=True)

        return betas_df

    def construct_untrapped_beta_df( self, beta_position, beta_direction, beta_energy, beta_num):
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

        #center_theta = sc.theta_center( initial_zpos, rho_center, initial_theta, self.config.trap_profile)

        # Use trapped_initial_theta to determine if trapped.
        #trapped_initial_theta = sc.min_theta( rho_center, initial_zpos, self.config.trap_profile)

        #max_radius = sc.max_radius( beta_energy, rho_center, self.config.trap_profile)

        track_properties = {
            # Conserved Quantities
            "hamiltonian": hamiltonian, #H = E + V (where E = γmc**2. H = E if no electric potential)
            "magnetic_moment": magnetic_moment,
            # Initial Kinematic Properties
            "start_energy": beta_energy, #note this is kinetic energy
            "start_gamma": sc.gamma(beta_energy),
            "start_rho_pos": initial_rho_pos,
            "start_phi_pos": initial_phi_pos,
            "start_zpos": initial_zpos,
            "start_theta": initial_theta,
            "start_cos_theta": np.cos(initial_theta / RAD_TO_DEG),
            "start_phi_dir": initial_phi_dir,
            "start_field": initial_field,
            "start_radius": initial_radius,
            "start_guiding_center_x": center_x,
            "start_guiding_center_y": center_y,
            "start_guiding_center_rho": rho_center,
            #"trapped_initial_theta": trapped_initial_theta,
            #"center_theta": center_theta,
            #"cos_center_theta": np.cos(center_theta / RAD_TO_DEG),
            #"max_radius": max_radius,
            #Computed Properties
            "trapped": False,
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
            #Final Kinematic Properties
            "end_energy": 0.0,
            "end_freq": 0.0,
            "end_time": np.nan,
            #Event IDs
            "track_num": 0,
            "beta_num": beta_num,
            "acq_num": np.nan,
            "trap_acq_num": np.nan,
        }

        beta_df = pd.DataFrame(track_properties, index=[beta_num])

        return beta_df
