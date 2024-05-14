""" simulation_blocks

This module contains all of the simulation "blocks" used by the 
Simulation class (see simulation.py). Each block simulates the action of
a concrete part of the pipeline from beta creation to a .spec file being
 written to disk by the ROACH DAQ. The module also contains the Config
class that reads the JSON config file and holds all of the configurable
parameters as well as the field profile. An instance of  the Config
class linked to a specific JSON config file is passed to each simulation
block.


The general approach is that pandas dataframes, each row describing a
single CRES data object (event, segment,  band, or track), are passed between
the blocks, each block adding complexity to the simulation. This general
structure is broken by the last class (Daq),
which is responsible for creating the .spec (binary) file output. This
.spec file can then be fed into Katydid just as real data would be.

Classes contained in module: 

    * DotDict
    * Config
    * Physics
    * EventBuilder
    * SegmentBuilder
    * BandBuilder
    * TrackBuilder
    * DMTrackBuilder
    * Daq

"""

import json
import math
import os
import pathlib
import yaml

import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
from time import process_time

from he6_cres_spec_sims.daq.frequency_domain_packet import FDpacket
from he6_cres_spec_sims.spec_tools.trap_field_profile import TrapFieldProfile
from he6_cres_spec_sims.spec_tools.beta_source.beta_source import BetaSource
import he6_cres_spec_sims.spec_tools.spec_calc.spec_calc as sc
import he6_cres_spec_sims.spec_tools.spec_calc.power_calc as pc

# TODO: Make the seed a config parameter, and pass rng(seed) around.

rng = default_rng()

# Math constants.

PI = math.pi
RAD_TO_DEG = 180 / math.pi
P11_PRIME = 1.84118  # First zero of J1 prime (bessel function)

# Physics constants.

ME = 5.10998950e5  # Electron rest mass (eV).
M = 9.1093837015e-31  # Electron rest mass (kg).
Q = 1.602176634e-19  # Electron charge (Coulombs).
C = 299792458  # Speed of light in vacuum (m/s)
J_TO_EV = 6.241509074e18  # Joule-ev conversion


class DotDict(dict):
    """Provides dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config:
    """
    TODO: Add a default value for each of these. The dictionary gets overwritten.


    A class used to contain the field map and configurable parameters
    associated with a given simulation configuration file (for example:
    config_example.json).

    ...

    Attributes
    ----------
    simulation, physics, eventbuilder, ... : DotDict
        A dictionary containing the configurable parameters associated
        with a given simulation block. The parameters can be accessed
        with dot.notation. For example eventbuilder.main_field would
        return a field value in T.

    trap_profile: Trap_profile
        An instance of a Trap_profile that corresponds to the main_field
        and trap_strength specified in the config file. Many of the
        spec_tool.spec_calc functions take the trap_profile as a
        parameter.

    field_strength: Trap_profile instance method
        Quick access to field strength values. field_strength(rho,z)=
        field magnitude in T at position (rho,z). Note that there is no
        field variation in phi.

    Methods
    -------
    load_config_file(config_filename)
        Loads the config file.

    load_field_profile()
        Loads the field profile.
    """

    def __init__(self, config_path, load_field=True, daq_only=False):
        """
        Parameters
        ----------
        config_filename: str
            The name of the config file contained in the
            he6_cres_spec_sims/config_files directory.
        """

        # Attributes:
        self.config_path = pathlib.Path(config_path)
        self.load_field = load_field
        self.daq_only = daq_only

        self.load_config_file()
        if self.load_field:
            self.load_field_profile()

    def load_config_file(self):
        """Loads the YAML config file and creates attributes associated
        with all configurable parameters.

        Parameters
        ----------
        config_filename: str
            The name of the config file contained in the
            he6_cres_spec_sims/config_files directory.

        Raises
        ------
        Exception
            If config file isn't found or can't be opened.
        """

        try:
            with open(self.config_path, "r") as read_file:
                config_dict = yaml.load(read_file, Loader=yaml.FullLoader)

                if self.daq_only:
                    self.daq = DotDict(config_dict["Daq"])

                else:
                    # Take config parameters from config_file.
                    self.settings = DotDict(config_dict["Settings"])
                    self.physics = DotDict(config_dict["Physics"])
                    self.eventbuilder = DotDict(config_dict["EventBuilder"])
                    self.segmentbuilder = DotDict(config_dict["SegmentBuilder"])
                    self.bandbuilder = DotDict(config_dict["BandBuilder"])
                    self.trackbuilder = DotDict(config_dict["TrackBuilder"])
                    self.downmixer = DotDict(config_dict["DMTrackBuilder"])
                    self.daq = DotDict(config_dict["Daq"])

        except Exception as e:
            print("Config file failed to load.")
            raise e

    def load_field_profile(self):
        """Uses the he6 trap geometry (2021), specified in the
        load_he6_trap module, and the main_field and trap strength
        specified in the config file to create an instance of
        Trap_profile.

        Parameters
        ----------
        None

        Raises
        ------
        Exception
            If field profile fails to load.
        """

        try:
            main_field = self.eventbuilder.main_field
            trap_current = self.eventbuilder.trap_current

            self.trap_profile = TrapFieldProfile(main_field, trap_current)
            self.field_strength = self.trap_profile.field_strength

        except Exception as e:
            print("Field profile failed to load.")
            raise e


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

    def generate_beta_position_direction(self, beta_num):

        # Could maybe improve this by not generating a new one each time,
        # it could be vectorized the way the energy is...

        position, direction = sc.random_beta_generator(
            self.config.physics, self.config.settings.rand_seed + beta_num
        )

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



class EventBuilder:
    """  Constructs a list of betas which are trapped within the detector volume
         (Doesn't hit waveguide walls && pitch angle is magnetically trapped)
    """

    def __init__(self, config):

        self.config = config
        self.physics = Physics(config)

    def run(self):

        print("~~~~~~~~~~~~EventBuilder Block~~~~~~~~~~~~~~\n")
        print("Constructing a set of trapped events:")
        # event_num denotes the number of trapped electrons simulated.
        event_num = 0
        # beta_num denotes the total number of betas produced in the trap.
        beta_num = 0

        # if simulating full daq we instead use the beta monitor rate to determine the number of events we should be seeing
        if self.config.settings.sim_daq==True:
            events_to_simulate = self.physics.number_of_events()
            betas_to_simulate = np.inf
        else:
            events_to_simulate = self.config.physics.events_to_simulate
            betas_to_simulate = self.config.physics.betas_to_simulate

            if events_to_simulate == -1:
                events_to_simulate = np.inf
            if betas_to_simulate == -1:
                betas_to_simulate = np.inf

        print(
            f"Simulating: num_events:{events_to_simulate}, num_betas:{betas_to_simulate}"
        )

        while (event_num < events_to_simulate) and (beta_num < betas_to_simulate):

            # print("\nEvent: {}/{}...\n".format(event_num, events_to_simulate - 1))

            # generate trapped beta
            is_trapped = False

            while not is_trapped and beta_num < betas_to_simulate:

                if beta_num % 250 == 0:
                    print(
                        f"\nBetas: {beta_num}/{betas_to_simulate - 1} simulated betas."
                    )
                    print(
                        f"\nEvents: {event_num}/{events_to_simulate-1} trapped events."
                    )

                # Does this miss some betas??? Be sure it doesn't.
                beta_num += 1

                (
                    initial_position,
                    initial_direction,
                ) = self.physics.generate_beta_position_direction(beta_num)

                energy = self.physics.generate_beta_energy(beta_num)

                single_segment_df = self.construct_untrapped_segment_df(
                    initial_position, initial_direction, energy, event_num, beta_num
                )

                is_trapped = self.trap_condition(single_segment_df)

            if event_num == 0:

                trapped_event_df = single_segment_df

            elif beta_num == betas_to_simulate:
                break

            else:
                trapped_event_df = pd.concat([trapped_event_df, single_segment_df], ignore_index=True)

            event_num += 1
        return trapped_event_df

    def construct_untrapped_segment_df(
        self, beta_position, beta_direction, beta_energy, event_num, beta_num
    ):
        """TODO:Document"""
        # Initial beta position and direction.
        initial_rho_pos = beta_position[0]
        initial_phi_pos = beta_position[1]
        initial_zpos = beta_position[2]

        initial_theta = beta_direction[0]
        initial_phi_dir = beta_direction[1]

        initial_field = self.config.field_strength(initial_rho_pos, initial_zpos)
        initial_radius = sc.cyc_radius(beta_energy, initial_field, initial_theta)

        # TODO: These center_x and center_y are a bit confusing. May
        # be nice to just build this into the power calc.
        center_x = initial_rho_pos - initial_radius * np.cos(
            (90 - initial_phi_dir) / RAD_TO_DEG
        )
        center_y = initial_radius * np.sin((90 - initial_phi_dir) / RAD_TO_DEG)

        rho_center = np.sqrt(center_x**2 + center_y**2)

        center_theta = sc.theta_center(
            initial_zpos, rho_center, initial_theta, self.config.trap_profile
        )

        # Use trapped_initial_theta to determine if trapped.
        trapped_initial_theta = sc.min_theta(
            rho_center, initial_zpos, self.config.trap_profile
        )
        max_radius = sc.max_radius(
            beta_energy, center_theta, rho_center, self.config.trap_profile
        )

        min_radius = sc.min_radius(
            beta_energy, center_theta, rho_center, self.config.trap_profile
        )

        segment_properties = {
            "energy": beta_energy,
            "gamma": sc.gamma(beta_energy),
            "energy_stop": 0.0,
            "initial_rho_pos": initial_rho_pos,
            "initial_phi_pos": initial_phi_pos,
            "initial_zpos": initial_zpos,
            "initial_theta": initial_theta,
            "cos_initial_theta": np.cos(initial_theta / RAD_TO_DEG),
            "initial_phi_dir": initial_phi_dir,
            "center_theta": center_theta,
            "cos_center_theta": np.cos(center_theta / RAD_TO_DEG),
            "initial_field": initial_field,
            "initial_radius": initial_radius,
            "center_x": center_x,
            "center_y": center_y,
            "rho_center": rho_center,
            "trapped_initial_theta": trapped_initial_theta,
            "max_radius": max_radius,
            "min_radius": min_radius,
            "avg_cycl_freq": 0.0,
            "b_avg": 0.0,
            "freq_stop": 0.0,
            "zmax": 0.0,
            "axial_freq": 0.0,
            "mod_index": 0.0,
            "segment_power": 0.0,
            "slope": 0.0,
            "segment_length": 0.0,
            "band_power_start": np.NaN,
            "band_power_stop": np.NaN,
            "band_num": np.NaN,
            "segment_num": 0,
            "event_num": event_num,
            "beta_num": beta_num,
            "fraction_of_spectrum": self.physics.bs.fraction_of_spectrum,
            "energy_accept_high": self.physics.bs.energy_acceptance_high,
            "energy_accept_low": self.physics.bs.energy_acceptance_low,
            "gamma_accept_high": sc.gamma(self.physics.bs.energy_acceptance_high),
            "gamma_accept_low": sc.gamma(self.physics.bs.energy_acceptance_low),
        }

        segment_df = pd.DataFrame(segment_properties, index=[event_num])

        return segment_df

    def trap_condition(self, segment_df):
        """TODO:Document"""
        segment_df = segment_df.reset_index(drop=True)

        if segment_df.shape[0] != 1:
            raise ValueError("trap_condition(): Input segment not a single row.")

        initial_theta = segment_df["initial_theta"][0]
        trapped_initial_theta = segment_df["trapped_initial_theta"][0]
        rho_center = segment_df["rho_center"][0]
        max_radius = segment_df["max_radius"][0]
        energy = segment_df["energy"][0]

        trap_condition = 0

        if initial_theta < trapped_initial_theta:
            # print("Not Trapped: Pitch angle too small.")
            trap_condition += 1

        if rho_center + max_radius > self.config.eventbuilder.decay_cell_radius:
            # print("Not Trapped: Collided with guide wall.")
            trap_condition += 1

        if trap_condition == 0:
            # print("Trapped!")
            return True
        else:
            return False


class SegmentBuilder:
    """ Constructs a list of tracks/ segments (interrupted by scatters) making up the trapped event
    """

    def __init__(self, config):

        self.config = config
        self.eventbuilder = EventBuilder(config)

    def run(self, trapped_event_df):
        """TODO: DOCUMENT"""
        print("~~~~~~~~~~~~SegmentBuilder Block~~~~~~~~~~~~~~\n")
        # Empty list to be filled with scattered segments.
        scattered_segments_list = []

        for event_index, event in trapped_event_df.iterrows():
            if event_index % 25 == 0:
                print("\nScattering Event :", event_index)

            # Assign segment 0 of event with a segment_length.
            event["segment_length"] = self.segment_length()

            # Fill the event with computationally intensive properties.
            event = self.fill_in_properties(event)

            # Extract position and center theta from event.
            center_x, center_y = event["center_x"], event["center_y"]
            rho_pos = event["initial_rho_pos"]
            phi_pos = event["initial_phi_pos"]
            zpos = 0
            center_theta = event["center_theta"]
            phi_dir = event["initial_phi_dir"]

            # Extract necessary parameters from event.
            # TODO(byron): Note that it is slightly incorrect to assume the power doesn't change as time passes.
            energy = event["energy"]
            energy_stop = event["energy_stop"]
            event_num = event["event_num"]
            beta_num = event["beta_num"]

            segment_radiated_power = event["segment_power"] * 2

            # Append segment 0 to scattered_segments_list because segment 0 is trapped by default.
            scattered_segments_list.append(event.values.tolist())

            # Begin with trapped beta (segment 0 of event).
            is_trapped = True
            jump_num = 0

            # The loop breaks when the trap condition is False or the jump_num exceeds self.jump_num_max.
            # This forces us to check to see that the current beta is trapped even when we don't want any scattering.
            # TODO: Improve the above issue. This will greatly improve run times.
            while True:

                if jump_num >= self.config.segmentbuilder.jump_num_max:
                    # print(
                    #     "Event reached jump_num_max : {}".format(
                    #         self.config.segmentbuilder.jump_num_max
                    #     )
                    # )
                    break

                print("Jump: {jump_num}".format(jump_num=jump_num))
                scattered_segment = event.copy()

                # Physics happens. TODO: This could maybe be wrapped into a different method.

                # Jump Size: Sampled from normal dist.
                mu = self.config.segmentbuilder.jump_size_eV
                sigma = self.config.segmentbuilder.jump_std_eV
                jump_size_eV = np.random.normal(mu, sigma)

                # Delta Pitch Angle: Sampled from normal dist.
                mu, sigma = 0, self.config.segmentbuilder.pitch_angle_costheta_std
                rand_float = np.random.normal(
                    mu, sigma
                )  # Necessary to properly distribute angles on a sphere.
                delta_center_theta = (np.arccos(rand_float) - PI / 2) * RAD_TO_DEG

                # Second, calculate new pitch angle and energy.
                # New Pitch Angle:
                center_theta = center_theta + delta_center_theta

                # Solving an issue caused by pitch angles larger than 90.
                if center_theta > 90:
                    center_theta = 180 - center_theta

                # New energy:
                energy = energy_stop - jump_size_eV

                # New position and direction. Only center_theta is changing right now.
                beta_position, beta_direction = (
                    [rho_pos, phi_pos, zpos],
                    [center_theta, phi_dir],
                )

                # Third, construct a scattered, meaning potentially not-trapped, segment df
                scattered_segment_df = self.eventbuilder.construct_untrapped_segment_df(
                    beta_position, beta_direction, energy, event_num, beta_num
                )

                # Fourth, check to see if the scattered beta is trapped.
                is_trapped = self.eventbuilder.trap_condition(scattered_segment_df)

                jump_num += 1

                # If the event is not trapped or the max number of jumps has been reached,
                # we do not want to write the df to the scattered_segments_list.
                if not is_trapped:
                    print("Event no longer trapped.")
                    break
                # if jump_num > self.config.segmentbuilder.jump_num_max:
                #     print(
                #         "Event reached jump_num_max : {}".format(
                #             self.config.segmentbuilder.jump_num_max
                #         )
                #     )
                #     break

                scattered_segment_df["segment_num"] = jump_num
                scattered_segment_df["segment_length"] = self.segment_length()
                scattered_segment_df = self.fill_in_properties(scattered_segment_df)

                scattered_segments_list.append(
                    scattered_segment_df.iloc[0].values.tolist()
                )

                # reset energy_stop, so that the next segment can be scattered based on this energy.
                energy_stop = scattered_segment_df["energy_stop"]

        scattered_df = pd.DataFrame(
            scattered_segments_list, columns=trapped_event_df.columns
        )

        return scattered_df

    def fill_in_properties(self, incomplete_scattered_segments_df):

        """DOCUMENT LATER"""

        df = incomplete_scattered_segments_df.copy()
        trap_profile = self.config.trap_profile
        main_field = self.config.eventbuilder.main_field
        decay_cell_radius = self.config.eventbuilder.decay_cell_radius

        # Calculate all relevant segment parameters. Order matters here.
        axial_freq = sc.axial_freq(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )

        # TODO: Make this more accurate as per discussion with RJ.
        b_avg = sc.b_avg(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )
        avg_cycl_freq = sc.energy_to_freq(df["energy"], b_avg)

        zmax = sc.max_zpos(
            df["energy"], df["center_theta"], df["rho_center"], trap_profile
        )
        mod_index = sc.mod_index(avg_cycl_freq, zmax)
        segment_radiated_power_te11 = (
            pc.power_calc(
                df["center_x"],
                df["center_y"],
                avg_cycl_freq,
                main_field,
                decay_cell_radius,
            )
            * 2
        )

        segment_radiated_power_tot = sc.power_larmor(main_field, avg_cycl_freq)

        # slope = sc.df_dt(
        #     df["energy"], self.config.eventbuilder.main_field, segment_radiated_power
        # )

        energy_stop = (
            df["energy"] - segment_radiated_power_tot * df["segment_length"] * J_TO_EV
        )

        # Replace negative energies if energy_stop is a float or pandas series
        if isinstance(energy_stop, pd.core.series.Series):
            energy_stop[energy_stop < 0]  = 1e-10
        elif energy_stop < 0:
            energy_stop = 1e-10

        freq_stop = sc.avg_cycl_freq(
            energy_stop, df["center_theta"], df["rho_center"], trap_profile
        )
        slope = (freq_stop - avg_cycl_freq) / df["segment_length"]

        segment_power = segment_radiated_power_te11 / 2

        df["axial_freq"] = axial_freq
        df["avg_cycl_freq"] = avg_cycl_freq
        df["b_avg"] = b_avg
        df["freq_stop"] = freq_stop
        df["energy_stop"] = energy_stop
        df["zmax"] = zmax
        df["mod_index"] = mod_index
        df["slope"] = slope
        df["segment_power"] = segment_power

        return df

    def segment_length(self):
        """TODO: DOCUMENT"""
        mu = self.config.segmentbuilder.mean_track_length
        segment_length = np.random.exponential(mu)

        return segment_length


class BandBuilder:
    """ Constructs list of sidebands and powers for trapped "segments", between scatters
    """

    def __init__(self, config):

        self.config = config

    def run(self, segments_df):

        print("~~~~~~~~~~~~BandBuilder Block~~~~~~~~~~~~~~\n")
        sideband_num = self.config.bandbuilder.sideband_num
        magnetic_modulation = self.config.bandbuilder.magnetic_modulation
        harmonic_sidebands = self.config.bandbuilder.harmonic_sidebands

        frac_total_segment_power_cut = (
            self.config.bandbuilder.frac_total_segment_power_cut
        )
        total_band_num = sideband_num * 2 + 1

        band_list = []

        for segment_index, row in segments_df.iterrows():

            if harmonic_sidebands:
                sideband_amplitudes = sc.sideband_calc(
                    row["energy"],
                    row["rho_center"],
                    row["avg_cycl_freq"],
                    row["axial_freq"],
                    row["zmax"],
                    self.config.trap_profile,
                    magnetic_modulation=magnetic_modulation,
                    num_sidebands=sideband_num,
                )[0]
            else:
                sideband_amplitudes = sc.anharmonic_sideband_calc(
                    row["energy"],
                    row["center_theta"],
                    row["rho_center"],
                    row["avg_cycl_freq"],
                    row["axial_freq"],
                    row["zmax"],
                    self.config.trap_profile,
                    magnetic_modulation=magnetic_modulation,
                    num_sidebands=sideband_num,
                )[0]

            for i, band_num in enumerate(range(-sideband_num, sideband_num + 1)):

                if sideband_amplitudes[i][1] < frac_total_segment_power_cut:
                    continue
                else:
                    # copy segment in order to fill in band specific values
                    row_copy = row.copy()

                    # fill in new avg_cycl_freq, band_power, band_num
                    # TODO: properly determine band power stop.
                    row_copy["avg_cycl_freq"] = sideband_amplitudes[i][0]
                    # Note that the sideband amplitudes need to be squared to give power.
                    row_copy["band_power_start"] = (
                        sideband_amplitudes[i][1] ** 2 * row.segment_power
                    )
                    row_copy["band_power_stop"] = row_copy["band_power_start"]
                    row_copy["band_num"] = band_num

                    # append to band_list, as it's better to grow a list than a df
                    band_list.append(row_copy.tolist())

        bands_df = pd.DataFrame(band_list, columns=segments_df.columns)

        return bands_df


class TrackBuilder:
    """ Assigns track truth parameters (e.g. start/ end times, frequencies) to created tracks
    """

    def __init__(self, config):

        self.config = config

    def run(self, bands_df):

        print("~~~~~~~~~~~~TrackBuilder Block~~~~~~~~~~~~~~\n")
        run_length = self.config.trackbuilder.run_length
        # events_to_simulate = self.config.physics.events_to_simulate
        events_simulated = int(bands_df["event_num"].max() + 1)
        print("events simulated: ", events_simulated)
        # TODO: Event timing is not currently physical. 
        # RJ: working on making this more physical, currently assuming beta monitor rate is equivalent to decay cell rate
        # Add time/freq start/stop.
        tracks_df = bands_df.copy()
        tracks_df["time_start"] = np.NaN
        tracks_df["time_stop"] = np.NaN
        tracks_df["file_in_acq"] = np.NaN

        tracks_df["freq_start"] = bands_df["avg_cycl_freq"]
        tracks_df["freq_stop"] = (
            bands_df["slope"] * bands_df["segment_length"] + bands_df["avg_cycl_freq"]
        )

        # dealing with timing of the events.
        # for now just put all events in the window... need to think about this.
        window = self.config.daq.n_files*self.config.daq.spec_length
        trapped_event_start_times = np.random.uniform(0, window, events_simulated)

        # iterate through the segment zeros and fill in start times.

        for index, row in bands_df[bands_df["segment_num"] == 0.0].iterrows():
            #             print(index)
            event_num = int(tracks_df["event_num"][index])
            #             print(event_num)
            file_num = int(trapped_event_start_times[event_num] // self.config.daq.spec_length)
            tracks_df["time_start"][index] = trapped_event_start_times[event_num] - self.config.daq.spec_length*file_num
            tracks_df["file_in_acq"][index] = file_num

        for event in range(0, events_simulated):

            # find max segment_num for each event
            segment_num_max = int(
                bands_df[bands_df["event_num"] == event]["segment_num"].max()
            )

            for segment in range(1, segment_num_max + 1):

                fill_condition = (tracks_df["event_num"] == float(event)) & (
                    tracks_df["segment_num"] == segment
                )
                previous_time_condition = (
                    (tracks_df["event_num"] == event)
                    & (tracks_df["segment_num"] == segment - 1)
                    & (tracks_df["band_num"] == 0.0)
                )
                #                 print("previous_time_condition : ", previous_time_condition)
                previous_segment_time_start = tracks_df[previous_time_condition][
                    "time_start"
                ].iloc[0]
                previous_segment_length = tracks_df[previous_time_condition][
                    "segment_length"
                ].iloc[0]

                for index, row in tracks_df[fill_condition].iterrows():
                    tracks_df["time_start"][index] = (
                        previous_segment_time_start + previous_segment_length
                    )

        tracks_df["time_stop"] = tracks_df["time_start"] + tracks_df["segment_length"]

        # tracks_df = tracks_df.drop(
        #     columns=[
        #         "initial_rho_pos",
        #         "initial_zpos",
        #         "initial_theta",
        #         "trapped_initial_theta",
        #         "initial_phi_dir",
        #         "center_theta",
        #         "initial_field",
        #         "initial_radius",
        #         "center_x",
        #         "center_y",
        #         "rho_center",
        #         "max_radius",
        #         "zmax",
        #         "mod_index",
        #         "avg_cycl_freq",
        #         "axial_freq",
        #     ]
        # )

        return tracks_df


class DMTrackBuilder:

    """ Downmixes freq_start and freq_stop of simulated tracks to observed frequency band out of DAQ
    """

    def __init__(self, config):

        self.config = config

    def run(self, tracks_df):
        """TODO:Document"""

        print("~~~~~~~~~~~~DMTrackBuilder Block~~~~~~~~~~~~~~\n")
        print(
            "DownMixing the cyclotron frequency with a {} GHz signal".format(
                np.around(self.config.downmixer.mixer_freq * 1e-9, 4)
            )
        )
        mixer_freq = self.config.downmixer.mixer_freq

        downmixed_tracks_df = tracks_df.copy()
        downmixed_tracks_df["freq_start"] = (
            downmixed_tracks_df["freq_start"] - mixer_freq
        )
        downmixed_tracks_df["freq_stop"] = downmixed_tracks_df["freq_stop"] - mixer_freq

        return downmixed_tracks_df


class DAQ:
    """  If desired, passes through list of produced downmixed tracks through DAQ, producing fake .spec(k) files
         These can be passed through Katydid, identically to data
    """

    def __init__(self, config):

        self.config = config

        # DAQ parameters derived from the config parameters.
        self.delta_f = config.daq.freq_bw / config.daq.freq_bins
        self.delta_t = 1 / self.delta_f
        self.slice_time = self.delta_t * self.config.daq.roach_avg
        self.pts_per_fft = config.daq.freq_bins * 2
        self.freq_axis = np.linspace(
            0, self.config.daq.freq_bw, self.config.daq.freq_bins
        )

        self.antenna_z = 50  # Ohms

        self.slices_in_spec = int(
            config.daq.spec_length / self.delta_t / self.config.daq.roach_avg
        )
        # This block size is used to create chunks of spec file that don't overhwelm the ram.
        self.slice_block = int(50 * 32768 / config.daq.freq_bins)

        # Get date for building out spec file paths.
        self.date = pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S")

        # Grab the gain_noise csv. TODO: Document what this needs to look like.
        self.gain_noise = pd.read_csv(self.config.daq.gain_noise_csv_path)

        # Divide the noise_mean_func by the roach_avg.
        # Need to add in U vs I side here.
        self.noise_mean_func = interpolate.interp1d(
            self.gain_noise.freq, self.gain_noise.noise_mean
        )
        self.gain_func = interpolate.interp1d(
            self.gain_noise.freq, self.gain_noise.gain
        )

        self.rng = np.random.default_rng(self.config.daq.rand_seed)

        self.noise_array = self.build_noise_floor_array()

    def run(self, downmixed_tracks_df):
        """
        This function is responsible for building out the spec files and calling the below methods.
        """
        self.tracks = downmixed_tracks_df
        self.build_results_dir()
        self.n_spec_files = downmixed_tracks_df.file_in_acq.nunique()
        self.build_spec_file_paths()
        self.build_empty_spec_files()

        # Define a random phase for each band.
        self.phase = self.rng.random(size=len(self.tracks))

        if self.config.daq.build_labels:
            self.build_label_file_paths()
            self.build_empty_label_files()

        for file_in_acq in range(self.n_spec_files):
            print(
                f"Building spec file {file_in_acq}. {self.config.daq.spec_length} s, {self.slices_in_spec} slices."
            )
            build_file_start = process_time()
            for i, start_slice in enumerate(
                np.arange(0, self.slices_in_spec, self.slice_block)
            ):

                # Iterate by the slice_block until you hit the end of the spec file.
                stop_slice = min(start_slice + self.slice_block, self.slices_in_spec)
                # This is the number of slices before averaging roach+avg slices together.
                num_slices = (stop_slice - start_slice) * self.config.daq.roach_avg

                noise_array = self.noise_array.copy()
                self.rng.shuffle(noise_array)
                noise_array = noise_array[: num_slices]

                signal_array = self.build_signal_chunk(
                    file_in_acq, start_slice, stop_slice
                )

                requant_gain_scaling = 2**self.config.daq.requant_gain

                spec_array = (
                    signal_array
                    * requant_gain_scaling
                    # LNA gain of 67dB
                    #TODO: make this a function of freq
                    *5e6
                    * self.gain_func(self.freq_axis)
                    + noise_array
                )

                spec_array = self.roach_slice_avg(spec_array)

                # Write chunk to spec file.
                self.write_to_spec(spec_array, self.spec_file_paths[file_in_acq])

                if self.config.daq.build_labels:
                    self.write_to_spec(
                        self.label_array, self.label_file_paths[file_in_acq]
                    )

            build_file_stop = process_time()
            print(
                f"Time to build file {file_in_acq}: {build_file_stop- build_file_start:.3f} s \n"
            )

        print("Done building spec files. ")
        return None

    def build_signal_chunk(self, file_in_acq, start_slice, stop_slice):

        print(f"file = {file_in_acq}, slices = [{start_slice}:{stop_slice}]")
        ith_slice = np.arange(
            start_slice * self.config.daq.roach_avg,
            stop_slice * self.config.daq.roach_avg,
        )
        num_slices = (stop_slice - start_slice) * self.config.daq.roach_avg

        slice_start = ith_slice * self.delta_t
        slice_stop = (ith_slice + 1) * self.delta_t

        # shape of t: (pts_per_fft, 1, num_slices). axis = 1 will be broadcast into the num_tracks.
        t = np.linspace(slice_start, slice_stop, self.pts_per_fft)
        t = np.expand_dims(t, axis=1)

        # shape of track info arrays: (num_tracks, num_slices)
        time_start = np.repeat(
            np.expand_dims(self.tracks.time_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        time_stop = np.repeat(
            np.expand_dims(self.tracks.time_stop.to_numpy(), axis=1), num_slices, axis=1
        )
        band_power_start = np.repeat(
            np.expand_dims(self.tracks.band_power_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        band_power_stop = np.repeat(
            np.expand_dims(self.tracks.band_power_stop.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        freq_start = np.repeat(
            np.expand_dims(self.tracks.freq_start.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )
        slope = np.repeat(
            np.expand_dims(self.tracks.slope.to_numpy(), axis=1), num_slices, axis=1
        )
        file_in_acq_array = np.repeat(
            np.expand_dims(self.tracks.file_in_acq.to_numpy(), axis=1),
            num_slices,
            axis=1,
        )

        # Reshape the random phase assigned to each band.
        band_phase = np.repeat(
            np.expand_dims(self.phase, axis=1),
            num_slices,
            axis=1,
        )

        # shape of slice_start/stop: (1, num_slices)
        slice_start = np.expand_dims(slice_start, axis=0)
        slice_stop = np.expand_dims(slice_stop, axis=0)

        # Note that you will get a division by zero warning if the time_stop and time_start are the same.
        band_powers = band_power_start + (band_power_stop - band_power_start) / (
            time_stop - time_start
        ) * (slice_start - time_start)

        # Caluculate max frequency within the signal in order to impose the LPF.
        freq_curr = freq_start + slope * (slice_stop - time_start)

        # shape of signal_alive_condition: (num_tracks, num_slices).
        # This condition is what imposes the LPF at the top of the freq bandwidth.
        signal_alive_condition = (
            (file_in_acq_array == file_in_acq)
            & (time_start <= slice_start)
            & (time_stop >= slice_stop)
            & (freq_curr <= self.config.daq.freq_bw)
        )

        # print(freq_curr[signal_alive_condition])
        # Setting all "dead" tracks to zero power.
        band_powers[~signal_alive_condition] = 0

        # Calculate voltage of signal.
        voltage = np.sqrt(band_powers * self.antenna_z)

        # The following condition selects only signals that are alive at some point during the time block.
        condition = band_powers.any(axis=1)

        # shape of signal_time_series: (pts_per_fft, num_tracks_alive_in_block, num_slices).
        # The below is the most time consuming operation and the array is very memory intensive.
        # What is happening is that the time domain signals for all slices in this block for each alive signal are made simultaneously
        # and then the signals are summed along axis =1 (track axis).
        # The factor of 2 is needed because the instantaeous frequency is the derivative
        # of the phase. The band_phase is a random phase assigned to each band.
        signal_time_series = voltage[condition, :] * np.sin(
            (
                freq_start[condition, :]
                + slope[condition, :] / 2 * (t - time_start[condition, :])
            )
            * (2 * np.pi * ((t - time_start[condition, :])))
            + (2 * np.pi * band_phase[condition, :])
        )

        if self.config.daq.build_labels:

            # shape of signal_time_series: (pts_per_fft, num_tracks_alive_in_block, num_slices).
            # Conduct a 1d FFT along axis = 0 (the time axis). This is less efficient than what is found in the
            # else statement but this is necessary to extract the label info.
            # shape of fft (pts_per_fft, num_tracks_alive_in_block, num_slices)
            fft = np.fft.fft(signal_time_series, axis=0, norm="ortho")[
                : self.pts_per_fft // 2
            ]
            fft = np.transpose(fft, (1, 0, 2))

            labels = np.abs(self.tracks.band_num.to_numpy()) + 1
            target = np.zeros((fft.shape[1:]))

            for i, alive_track_fft in enumerate(fft):

                # How to create this mask is a bit tricky. Not sure what factor to use.
                # This is harder than expected due to the natural fluctuations in bin power.
                # I'm not getting continuous masks. One idea is to make the mask condition column-wise...
                # Needs to be the magnitude!! Ok.
                # Keep the axis =0 max because this makes the labels robust against SNR fluctuations across the track.
                mask = (np.abs(alive_track_fft) ** 2) > (
                    np.abs(alive_track_fft) ** 2
                ).max(axis=0) / 10

                target[mask] = labels[condition][i]

            # Don't actually average or the labels will get weird. Just sample according to the roach_avg
            label_array = target.T[:: self.config.daq.roach_avg]

            self.label_array = label_array
            fft = fft.sum(axis=0)

        else:
            # shape of signal_time_series: (pts_per_fft, num_slices).
            signal_time_series = signal_time_series.sum(axis=1)

            # shape of signal_time_series: (pts_per_fft, num_slices). Conduct a 1d FFT along axis = 0 (the time axis).
            fft = np.fft.fft(signal_time_series, axis=0, norm="ortho")[
                : self.pts_per_fft // 2
            ]

        signal_array = np.real(fft)**2

        #can't average until we've added noise and gotten the powers. 

        return signal_array.T

    def write_to_spec(self, spec_array, spec_file_path):
        """
        Append to an existing spec file. This is necessary because the spec arrays get too large for 1s
        worth of data.
        """
        # print("Writing to file path: {}\n".format(spec_file_path))

        # Make spec file:
        slices_in_spec, freq_bins_in_spec = spec_array.shape

        zero_hdrs = np.zeros((slices_in_spec, 32))

        # Append empty (zero) headers to the spec array.
        spec_array_hdrs = np.hstack((zero_hdrs, spec_array))

        data = spec_array_hdrs.flatten().astype("uint8")

        # Pass "ab" to append to a binary file
        with open(spec_file_path, "ab") as spec_file:

            # Write data to spec_file.
            data.tofile(spec_file)

        return None

    def build_noise_floor_array(self):
        """
        Build a noise floor array with self.slice_block slices.
        Note that this only works for roach avg = 2 at this point.
        """

        self.freq_axis = np.linspace(
            0, self.config.daq.freq_bw, self.config.daq.freq_bins
        )

        delta_f_12 = 2.4e9 / 2**13

        noise_power_scaling = self.delta_f / delta_f_12
        requant_gain_scaling = (2**self.config.daq.requant_gain) / (2**self.config.daq.noise_file_gain)
        noise_scaling = noise_power_scaling * requant_gain_scaling

        # Chisquared noise:
        noise_array = self.rng.chisquare(
            df=2, size=(self.slice_block * self.config.daq.roach_avg, self.config.daq.freq_bins)
        )
        noise_array *= self.noise_mean_func(self.freq_axis) / noise_array.mean(axis=0)

        # Scale by noise power.
        noise_array *= noise_scaling
        noise_array = np.around(noise_array).astype("uint8")

        return noise_array

    def build_spec_file_paths(self):

        spec_file_paths = []
        for idx in range(self.n_spec_files):

            spec_path = self.spec_files_dir / "{}_spec_{}.spec".format(
                self.config.daq.spec_prefix, idx
            )
            spec_file_paths.append(spec_path)

        self.spec_file_paths = spec_file_paths

        return None

    def build_label_file_paths(self):

        label_file_paths = []
        for idx in range(self.n_spec_files):

            spec_path = self.label_files_dir / "{}_label_{}.spec".format(
                self.config.daq.spec_prefix, idx
            )
            label_file_paths.append(spec_path)

        self.label_file_paths = label_file_paths

        return None

    def build_results_dir(self):

        # First make a results_dir with the same name as the config.
        config_name = self.config.config_path.stem
        parent_dir = self.config.config_path.parents[0]

        self.results_dir = parent_dir / config_name

        # If results_dir doesn't exist, then create it.
        if not self.results_dir.is_dir():
            self.results_dir.mkdir()
            print("created directory : ", self.results_dir)

        self.spec_files_dir = self.results_dir / "spec_files"

        # If spec_files_dir doesn't exist, then create it.
        if not self.spec_files_dir.is_dir():
            self.spec_files_dir.mkdir()
            print("created directory : ", self.spec_files_dir)

        if self.config.daq.build_labels:

            self.label_files_dir = parent_dir / config_name / "label_files"

            # If spec_files_dir doesn't exist, then create it.
            if not self.label_files_dir.is_dir():
                self.label_files_dir.mkdir()
                print("created directory : ", self.label_files_dir)

        return None

    def build_empty_spec_files(self):
        """
        Build empty spec files to be filled with data or labels.
        """
        # Pass "wb" to write a binary file. But here we just build the files.
        for idx, spec_file_path in enumerate(self.spec_file_paths):
            with open(spec_file_path, "wb") as spec_file:
                pass

        return None

    def build_empty_label_files(self):
        """
        Build empty spec files to be filled with data or labels.
        """
        # Pass "wb" to write a binary file. But here we just build the files.
        for idx, spec_file_path in enumerate(self.label_file_paths):
            with open(spec_file_path, "wb") as spec_file:
                pass

        return None

    def build_labels(self):
        """
        This may need to just be a flag... Should I write these to spec files as well?
        One could imagine that then the preprocessing is to do a maxpool on everything as we read it in? Then build a smaller
        more manageable array that's still 1s worth of data. That could be nice.
        Should think about how to get the spec files into arrays. Maybe this should be a method of the results class?
        """
        return None

    def roach_slice_avg(self, signal_array):

        N = int(self.config.daq.roach_avg)

        if self.config.daq.roach_inverted_flag == True:
            result = signal_array[::N]

        else:
            if signal_array.shape[0] % 2 == 0:
                result = signal_array[1::2] + signal_array[::2]
            else:
                result = signal_array[1::2] + signal_array[:-1:2]

        return result

    def spec_to_array(
        self, spec_path, slices=-1, packets_per_slice=1, start_packet=None
    ):
        """
        TODO: Document.
        Making this just work for one packet per spectrum because that works for simulation in Katydid.
        * Make another function that works with 4 packets per spectrum (for reading the Kr data).
        """

        BYTES_IN_PAYLOAD = self.config.daq.freq_bins
        BYTES_IN_HEADER = 32
        BYTES_IN_PACKET = BYTES_IN_PAYLOAD + BYTES_IN_HEADER

        if slices == -1:
            spec_array = np.fromfile(spec_path, dtype="uint8", count=-1).reshape(
                (-1, BYTES_IN_PACKET)
            )[:, BYTES_IN_HEADER:]
        else:
            spec_array = np.fromfile(
                spec_path, dtype="uint8", count=BYTES_IN_PAYLOAD * slices
            ).reshape((-1, BYTES_IN_PACKET))[:, BYTES_IN_HEADER:]

        if packets_per_slice > 1:

            spec_flat_list = [
                spec_array[(start_packet + i) % packets_per_slice :: packets_per_slice]
                for i in range(packets_per_slice)
            ]
            spec_flat = np.concatenate(spec_flat_list, axis=1)
            spec_array = spec_flat

        print(spec_array.shape)

        return spec_array
